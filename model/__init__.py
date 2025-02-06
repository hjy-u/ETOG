from .segmenter import ETOG_res, ETOG_vit
from loguru import logger

def build_etog_segmenter_res(args):
    model = ETOG_res(args)
    backbone = []
    head = []
    fix = []
    backbone_names = []
    head_names = []
    fix_names = []
    for k, v in model.named_parameters():
        if (k.startswith('backbone') and 'positional_embedding' not in k or 'bridger' in k) and v.requires_grad:
        # if 'bridger' in k and v.requires_grad:
            backbone.append(v)
            backbone_names.append(k)
        elif v.requires_grad:
            head.append(v)
            head_names.append(k)
        else:
            fix.append(v)
            fix_names.append(k)
    logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [{
        'params': backbone,
        'initial_lr': args.lr_multi * args.base_lr
    }, {
        'params': head,
        'initial_lr': args.base_lr
    }]

    n_backbone_parameters = sum(p.numel() for p in backbone)
    logger.info(f'number of updated params (Backbone): {n_backbone_parameters}.')
    n_head_parameters = sum(p.numel() for p in head)
    logger.info(f'number of updated params (Head)    : {n_head_parameters}')
    n_fixed_parameters = sum(p.numel() for p in fix)
    logger.info(f'number of fixed params             : {n_fixed_parameters}')
    return model, param_list

def build_etog_segmenter_vit(args):
    model = ETOG_vit(args)
    backbone = []
    head = []
    fix = []
    backbone_names = []
    head_names = []
    fix_names = []
    for k, v in model.named_parameters():
        if (k.startswith('backbone') and 'positional_embedding' not in k or 'bridger' in k) and v.requires_grad:
        # if 'bridger' in k and v.requires_grad:
            backbone.append(v)
            backbone_names.append(k)
        elif v.requires_grad:
            head.append(v)
            head_names.append(k)
        else:
            fix.append(v)
            fix_names.append(k)
    logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [{
        'params': backbone,
        'initial_lr': args.lr_multi * args.base_lr
    }, {
        'params': head,
        'initial_lr': args.base_lr
    }]

    n_backbone_parameters = sum(p.numel() for p in backbone)
    logger.info(f'number of updated params (Backbone): {n_backbone_parameters}.')
    n_head_parameters = sum(p.numel() for p in head)
    logger.info(f'number of updated params (Head)    : {n_head_parameters}')
    n_fixed_parameters = sum(p.numel() for p in fix)
    logger.info(f'number of fixed params             : {n_fixed_parameters}')
    return model, param_list