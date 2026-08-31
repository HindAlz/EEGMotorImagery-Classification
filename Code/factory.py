from models import ATCNet, CSPLDA, EEGNet, ShallowConvNet

def build_model(
    name,
    config,
    n_classes,
    n_channels,
    n_samples,
):
    if name == "eegnet":
        return EEGNet(
            n_classes=n_classes,
            n_channels=n_channels,
            n_samples=n_samples,
            dropout=config["dropout"],
            kernel_length=config["kernel_length"],
            F1=config["F1"],
            D=config["D"],
            F2=config["F2"],
        )

    if name == "shallowconvnet":
        return ShallowConvNet(
            n_classes=n_classes,
            n_channels=n_channels,
            n_samples=n_samples,
            dropout=config["dropout"],
        )

    if name == "atcnet":
        return ATCNet(
            n_classes=n_classes,
            n_channels=n_channels,
            n_samples=n_samples,
            n_windows=config["n_windows"],
            attention=config.get("attention"),
            eegn_F1=config["eegn_F1"],
            eegn_D=config["eegn_D"],
            eegn_kernel_size=config["eegn_kernel_size"],
            eegn_pool_size=config["eegn_pool_size"],
            eegn_dropout=config["eegn_dropout"],
            tcn_depth=config["tcn_depth"],
            tcn_kernel_size=config["tcn_kernel_size"],
            tcn_filters=config["tcn_filters"],
            tcn_dropout=config["tcn_dropout"],
            tcn_activation=config["tcn_activation"],
            fuse=config["fuse"],
        )

    if name == "csp_lda":
        return CSPLDA(
            n_components=config["n_components"],
            reg=config.get("reg"),
            log=config["log"],
            norm_trace=config["norm_trace"],
        )

    raise ValueError(f"Unknown model: {name}")