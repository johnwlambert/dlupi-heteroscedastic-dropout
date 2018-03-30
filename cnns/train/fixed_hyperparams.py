
# John Lambert

import argparse

def get_fixed_hyperparams(parser):
    """ Hyperparameters that remain fixed across all experiments """
    parser.add_argument('--pred_logvar_domain', type=bool, default=True)
    parser.add_argument('--use_l2_sigma_reg', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='imagenet_bboxes')
    parser.add_argument('--dataset_path', type=str, default='')  # '/vision/group/CIFAR-10')
    parser.add_argument('--num_classes', type=int, default=1000)  # 10 for CIFAR-10
    parser.add_argument('--std', type=float, default=0.02, help='for weight')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_decay_every', default=5, type=int, help='decay by 10 every n epoch')  # or every 40
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--use_bn', type=bool, default=True)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--noise_mu', default=1.0, type=float)
    parser.add_argument('--image_size', type=int, default=224)  # 32 for cifar-10
    parser.add_argument('--num_epochs_tolerate_no_acc_improv', type=int, default=5)
    parser.add_argument('--use_identity_at_test_time', type=bool, default=True)
    parser.add_argument('--share_conv_layers', type=bool, default=True)
    parser.add_argument('--xstar_backprop_through_conv', type=bool, default=False)

    # purely for the localization task
    parser.add_argument('--new_localization_train_path', type=str, default='')
    parser.add_argument('--new_localization_val_path', type=str, default='')
    parser.add_argument('--new_localization_test_path', type=str, default='')

    parser.add_argument('--new_localization_annotation_path', type=str, default='')
    parser.add_argument('--num_imagenet_classes_to_use', type=int, default=1000)

    parser.add_argument('--fc_size', type=int, default=4096 ) # CHANGE BACK TO 20
    parser.add_argument('--momentum', default=0.9, type=float )
    parser.add_argument('--num_workers', type=int, default=20) # CHANGE BACK TO 20
    parser.add_argument('--ckpt_path', type=str, default='')

    parser.add_argument('--percent_of_xstar_to_use_in_train', type=float,
                        default=100.0 )# )3.92 ) #  ) # corresponds to 8,660 / 75k (* 100) to make it a percent

    parser.add_argument('--info_dropout_activation_fn', type=str, default='softplus') # 'relu'
    parser.add_argument('--sigma_regularization_multiplier', default=100.0, type=float)

    # Normally, we want this to be true, unless generating random values
    parser.add_argument('--use_python_random_seed', default=True, type=bool )
    parser.add_argument('--use_full_imagenet', default=False, type=bool )
    parser.add_argument('--percent_x_to_train_with', type=float, default=1.0) # 1.00 means 100%
    parser.add_argument('--num_dropout_computations', type=int, default=2 )
    parser.add_argument('--xstar_loss_multiplier', default=0.0, type=float)
    parser.add_argument('--num_channels', type=int, default=3) # just use 1 for CIFAR-10
    parser.add_argument('--use_dcgan_autoencoder', type=bool, default=False)
    parser.add_argument('--start_num_encoder_filt', type=int, default=64)
    parser.add_argument('--end_num_decoder_filt', type=int, default=64)
    parser.add_argument('--max_grad_norm', type=float, default=10)
    return parser