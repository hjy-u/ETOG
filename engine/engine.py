import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from loguru import logger
from utils.dataset import tokenize, overlay_davis
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather,
                        trainMetricGPU)


#before run, plz check thresh value

def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, iou_meter, pr_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    thresh = 0.5 #if 'clipfuse' in args.exp_name else 0.35

    model.train()
    time.sleep(2)
    end = time.time()

    for i, (image, text, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # data
        image = image.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).unsqueeze(1)
        # if image_masked is not None:
        #     image_masked = image_masked.cuda(non_blocking=True)
        # neg_text = None #neg_text.cuda(non_blocking=True)

        # forward
        with amp.autocast():
            # if neg_text is None:
            pred, target, loss = model(image, text, target)
            # else:
            #     pred, target, loss = model(image, text, target, neg_text)


        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        # metric
        iou, pr5 = trainMetricGPU(pred, target, thresh, 0.5)
        dist.all_reduce(loss.detach())
        dist.all_reduce(iou)
        dist.all_reduce(pr5)
        loss = loss / dist.get_world_size()
        iou = iou / dist.get_world_size()
        pr5 = pr5 / dist.get_world_size()

        loss_meter.update(loss.item(), image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            if dist.get_rank() in [-1, 0]:
                wandb.log(
                    {
                        "time/batch": batch_time.val,
                        "time/data": data_time.val,
                        "training/lr": lr.val,
                        "training/loss": loss_meter.val,
                        "training/iou": iou_meter.val,
                        "training/prec@50": pr_meter.val,
                    },
                    step=epoch * len(train_loader) + (i + 1))


@torch.no_grad()
def validate(val_loader, model, epoch, args):
    iou_list = []
    thresh = 0.5 
    model.eval()
    time.sleep(2)
    I = []
    U = []
    # cum_inter = 0
    # cum_union = 0
    for imgs, texts, param in val_loader:
        # data
        imgs = imgs.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)
        # inference
        preds, _ = model(imgs, texts)
        preds = torch.sigmoid(preds)
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(preds,
                                  size=imgs.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True)
        preds = preds.squeeze(1)
        # process one batch
        for pred, mask_dir, mat, ori_size in zip(preds, param['mask_dir'],
                                                 param['inverse'],
                                                 param['ori_size']):
            h, w = np.array(ori_size)
            mat = np.array(mat)
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            pred = np.array(pred > thresh)
            mask = cv2.imread(mask_dir, flags=cv2.IMREAD_GRAYSCALE)
            mask = mask / 255.
            # iou
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            I.append(np.sum(inter))
            U.append(np.sum(union))
            # cum_inter += np.sum(inter)
            # cum_union += np.sum(union)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(imgs.device)
    iou_list = concat_all_gather(iou_list)
    I = np.stack(I)
    I = torch.from_numpy(I).to(imgs.device)
    I = concat_all_gather(I).sum()

    U = np.stack(U)
    U = torch.from_numpy(U).to(imgs.device)
    U = concat_all_gather(U).sum()
    oIoU = I / U
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    # oiou = cum_inter/cum_union
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}  oIoU:{:.2f}'.format(
        epoch, args.epochs, 100. * iou.item(), 100 * oIoU)
    logger.info(head + temp)
    return iou.item(), prec


@torch.no_grad()
def inference(test_loader, model, args):
    iou_list = []
    thresh = 0.5
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    # cum_inter = 0
    # cum_union = 0
    I = []
    U = []
    model.eval()
    time.sleep(2)
    for img, param in tbar:
        # data
        img = img.cuda(non_blocking=True)
        mask = cv2.imread(param['mask_dir'][0], flags=cv2.IMREAD_GRAYSCALE)
        # dump image & mask
        if args.visualize:
            seg_id = param['seg_id'][0].cpu().numpy()
            img_name = '{}-img.jpg'.format(seg_id)
            mask_name = '{}-mask.png'.format(seg_id)
            cv2.imwrite(filename=os.path.join(args.vis_dir, img_name),
                        img=param['ori_img'][0].cpu().numpy())
            cv2.imwrite(filename=os.path.join(args.vis_dir, mask_name),
                        img=mask)
            mask_vis_name = '{}-mask_vis.png'.format(seg_id)
            new_mask = (mask / 255).astype(np.uint8)
            mask_vis = overlay_davis(param['ori_img'][0].cpu().numpy(), new_mask)
            cv2.imwrite(filename=os.path.join(args.vis_dir, mask_vis_name),
                        img=mask_vis)
        # multiple sentences
        for sent in param['sents']:
            mask = mask / 255.
            text = tokenize(sent, args.word_len, True)
            text = text.cuda(non_blocking=True)
            # inference
            pred, attn = model(img, text)
            pred = torch.sigmoid(pred)
            if pred.shape[-2:] != img.shape[-2:]:
                pred = F.interpolate(pred,
                                     size=img.shape[-2:],
                                     mode='bicubic',
                                     align_corners=True)
            pred = pred.squeeze()
            # process one sentence
            h, w = param['ori_size'].numpy()[0]
            mat = param['inverse'].numpy()[0]
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            pred = np.array(pred > thresh)
            # iou
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            I.append(np.sum(inter))
            U.append(np.sum(union))
            # cum_inter += np.sum(inter)
            # cum_union += np.sum(union)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
            # dump prediction
            if args.visualize:
                ori_img = param['ori_img'][0].cpu().numpy()
                attn_map = cv2.warpAffine(attn, mat, (w, h),
                                         flags=cv2.INTER_CUBIC,
                                         borderValue=0.)
                maps_vis = cv2.applyColorMap((attn_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                maps_vis = (0.5 * cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR) + 0.5 * maps_vis).astype(np.uint8)
                attn_name = '{}-iou={:.2f}-{}-attn.png'.format(seg_id, iou * 100, sent)
                cv2.imwrite(filename=os.path.join(args.vis_dir, attn_name),
                            img=maps_vis)

                pred = np.array(pred*255, dtype=np.uint8)
                sent = "_".join(sent[0].split(" "))
                pred_name = '{}-iou={:.2f}-{}.png'.format(seg_id, iou*100, sent)
                cv2.imwrite(filename=os.path.join(args.vis_dir, pred_name),
                            img=pred)
                mask_vis_name = '{}-iou={:.2f}-{}_maskvis.png'.format(seg_id, iou*100, sent)
                new_pred = (pred/255).astype(np.uint8)
                mask_vis = overlay_davis(param['ori_img'][0].cpu().numpy(), new_pred)
                cv2.imwrite(filename=os.path.join(args.vis_dir, mask_vis_name),
                            img=mask_vis)

    logger.info('=> Metric Calculation <=')
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(img.device)
    I = np.stack(I)
    I = torch.from_numpy(I).to(img.device)
    I = I.sum()

    U = np.stack(U)
    U = torch.from_numpy(U).to(img.device)
    U = U.sum()
    oIoU = I / U
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    # oiou = cum_inter/cum_union
    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres*10)
        value = prec_list[i].item()
        prec[key] = value
    logger.info('IoU={:.2f}'.format(100.*iou.item()))
    logger.info('oIoU={:.2f}'.format(100. * oIoU))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100.*v))

    return iou.item(), prec

@torch.no_grad()
def inference_vit(test_loader, model, args):
    iou_list = []
    thresh = 0.5
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    I = []
    U = []
    model.eval()
    time.sleep(2)
    for img, param in tbar:
        # data
        img = img.cuda(non_blocking=True)
        mask = cv2.imread(param['mask_dir'][0], flags=cv2.IMREAD_GRAYSCALE)
        if args.visualize:
            seg_id = param['seg_id'][0].cpu().numpy()
            img_name = '{}-img.jpg'.format(seg_id)
            mask_name = '{}-mask.png'.format(seg_id)
            cv2.imwrite(filename=os.path.join(args.vis_dir, img_name),
                        img=param['ori_img'][0].cpu().numpy())
            cv2.imwrite(filename=os.path.join(args.vis_dir, mask_name),
                        img=mask)
            mask_vis_name = '{}-mask_vis.png'.format(seg_id)
            new_mask = (mask / 255).astype(np.uint8)
            mask_vis = overlay_davis(param['ori_img'][0].cpu().numpy(), new_mask)
            cv2.imwrite(filename=os.path.join(args.vis_dir, mask_vis_name),
                        img=mask_vis)
        # multiple sentences
        for sent in param['sents']:
            mask = mask / 255.
            text = tokenize(sent, args.word_len, True)
            text = text.cuda(non_blocking=True)
            # inference
            pred, _ = model(img, text)
            pred = torch.sigmoid(pred)
            if pred.shape[-2:] != img.shape[-2:]:
                pred = F.interpolate(pred,
                                     size=img.shape[-2:],
                                     mode='bicubic',
                                     align_corners=True)
            pred = pred.squeeze()
            # process one sentence
            h, w = param['ori_size'].numpy()[0]
            mat = param['inverse'].numpy()[0]
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            pred = np.array(pred > thresh)
            # iou
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            I.append(np.sum(inter))
            U.append(np.sum(union))
            # cum_inter += np.sum(inter)
            # cum_union += np.sum(union)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
            # dump prediction
            if args.visualize:
                pred = np.array(pred*255, dtype=np.uint8)
                sent = "_".join(sent[0].split(" "))
                pred_name = '{}-iou={:.2f}-{}.png'.format(seg_id, iou*100, sent)
                cv2.imwrite(filename=os.path.join(args.vis_dir, pred_name),
                            img=pred)
                mask_vis_name = '{}-iou={:.2f}-{}_maskvis.png'.format(seg_id, iou*100, sent)
                new_pred = (pred/255).astype(np.uint8)
                mask_vis = overlay_davis(param['ori_img'][0].cpu().numpy(), new_pred)
                cv2.imwrite(filename=os.path.join(args.vis_dir, mask_vis_name),
                            img=mask_vis)

    logger.info('=> Metric Calculation <=')
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(img.device)
    I = np.stack(I)
    I = torch.from_numpy(I).to(img.device)
    I = I.sum()

    U = np.stack(U)
    U = torch.from_numpy(U).to(img.device)
    U = U.sum()
    oIoU = I / U
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    # oiou = cum_inter/cum_union
    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres*10)
        value = prec_list[i].item()
        prec[key] = value
    logger.info('IoU={:.2f}'.format(100.*iou.item()))
    logger.info('oIoU={:.2f}'.format(100. * oIoU))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100.*v))

    return iou.item(), prec