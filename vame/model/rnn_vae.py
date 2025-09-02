import torch
from torch import nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.tensorboard.writer import SummaryWriter
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Union, Optional
import datetime

from vame.model.dataloader import SEQUENCE_DATASET
from vame.model.rnn_model import RNN_VAE
from vame.schemas.states import TrainModelFunctionSchema, save_state
from vame.logging.logger import VameLogger, TqdmToLogger


logger_config = VameLogger(__name__)
logger = logger_config.logger
tqdm_to_logger = TqdmToLogger(logger)

# TensorBoard configuration (hardcoded)
TENSORBOARD_ENABLED = True
TENSORBOARD_LOG_FREQUENCY = 1  # Log every N batches
TENSORBOARD_LOG_HISTOGRAMS = False  # Set to True to log parameter histograms

# make sure torch uses cuda for GPU computing
use_gpu = torch.cuda.is_available()
if use_gpu:
    logger.info("GPU detected")
    logger.info(f"GPU used: {torch.cuda.get_device_name(0)}")
else:
    logger.info("No GPU found... proceeding with CPU (slow!)")
    torch.device("cpu")


def reconstruction_loss(
    x: torch.Tensor,
    x_tilde: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """
    Compute the reconstruction loss between input and reconstructed data.

    Parameters
    ----------
    x : torch.Tensor
        Input data tensor.
    x_tilde : torch.Tensor
        Reconstructed data tensor.
    reduction : str
        Type of reduction for the loss.

    Returns
    -------
    torch.Tensor
        Reconstruction loss.
    """
    mse_loss = nn.MSELoss(reduction=reduction)
    rec_loss = mse_loss(x_tilde, x)
    return rec_loss


def future_reconstruction_loss(
    x: torch.Tensor,
    x_tilde: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """
    Compute the future reconstruction loss between input and predicted future data.

    Parameters
    ----------
    x : torch.Tensor
        Input future data tensor.
    x_tilde : torch.Tensor
        Reconstructed future data tensor.
    reduction : str
        Type of reduction for the loss.

    Returns
    -------
    torch.Tensor
        Future reconstruction loss.
    """
    mse_loss = nn.MSELoss(reduction=reduction)
    rec_loss = mse_loss(x_tilde, x)
    return rec_loss


def cluster_loss(
    H: torch.Tensor,
    kloss: int,
    lmbda: float,
    batch_size: int,
) -> torch.Tensor:
    """
    Compute the cluster loss.

    Parameters
    ----------
    H : torch.Tensor
        Latent representation tensor.
    kloss : int
        Number of clusters.
    lmbda : float
        Lambda value for the loss.
    batch_size : int
        Size of the batch.

    Returns
    -------
    torch.Tensor
        Cluster loss.
    """
    gram_matrix = (H.T @ H) / batch_size
    _, sv_2, _ = torch.svd(gram_matrix)
    sv = torch.sqrt(sv_2[:kloss])
    loss = torch.sum(sv)
    return lmbda * loss


def kullback_leibler_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Kullback-Leibler divergence loss.
    See Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 - https://arxiv.org/abs/1312.6114

    Formula: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Parameters
    ----------
    mu : torch.Tensor
        Mean of the latent distribution.
    logvar : torch.Tensor
        Log variance of the latent distribution.

    Returns
    -------
    torch.Tensor
        Kullback-Leibler divergence loss.
    """
    # I'm using torch.mean() here as the sum() version depends on the size of the latent vector
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def kl_annealing(
    epoch: int,
    kl_start: int,
    annealtime: int,
    function: str,
) -> float:
    """
    Anneal the Kullback-Leibler loss to let the model learn first the reconstruction of the data
    before the KL loss term gets introduced.

    Parameters
    ----------
    epoch : int
        Current epoch number.
    kl_start : int
        Epoch number to start annealing the loss.
    annealtime : int
        Annealing time.
    function : str
        Annealing function type.

    Returns
    -------
    float
        Annealed weight value for the loss.
    """
    if epoch > kl_start:
        if function == "linear":
            new_weight = min(1, (epoch - kl_start) / (annealtime))

        elif function == "sigmoid":
            new_weight = float(1 / (1 + np.exp(-0.9 * (epoch - annealtime))))
        else:
            raise NotImplementedError('currently only "linear" and "sigmoid" are implemented')

        return new_weight
    else:
        new_weight = 0
        return new_weight


def gaussian(
    ins: torch.Tensor,
    is_training: bool,
    seq_len: int,
    std_n: float = 0.8,
) -> torch.Tensor:
    """
    Add Gaussian noise to the input data.

    Parameters
    ----------
    ins : torch.Tensor
        Input data tensor.
    is_training : bool
        Whether it is training mode.
    seq_len : int
        Length of the sequence.
    std_n : float
        Standard deviation for the Gaussian noise.

    Returns
    -------
    torch.Tensor
        Noisy input data tensor.
    """
    if is_training:
        emp_std = ins.std(1) * std_n
        emp_std = emp_std.unsqueeze(2).repeat(1, 1, seq_len)
        emp_std = emp_std.permute(0, 2, 1)
        noise = Variable(ins.data.new(ins.size()).normal_(0, 1))
        return ins + (noise * emp_std)
    return ins


def train(
    train_loader: Data.DataLoader,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    anneal_function: str,
    BETA: float,
    kl_start: int,
    annealtime: int,
    seq_len: int,
    future_decoder: bool,
    future_steps: int,
    scheduler: Union[torch.optim.lr_scheduler._LRScheduler, ReduceLROnPlateau, StepLR],
    mse_red: str,
    mse_pred: str,
    kloss: int,
    klmbda: float,
    bsize: int,
    noise: bool,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
) -> Tuple[float, float, float, float, float, float, int]:
    """
    Train the model.

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader.
    epoch : int
        Current epoch number.
    model : nn.Module
        Model to be trained.
    optimizer : Optimizer
        Optimizer for training.
    anneal_function : str
        Annealing function type.
    BETA : float
        Beta value for the loss.
    kl_start : int
        Epoch number to start annealing the loss.
    annealtime : int
        Annealing time.
    seq_len : int
        Length of the sequence.
    future_decoder : bool
        Whether a future decoder is used.
    future_steps : int
        Number of future steps to predict.
    scheduler : Union[_LRScheduler, ReduceLROnPlateau]
        Learning rate scheduler.
    mse_red : str
        Reduction type for MSE reconstruction loss.
    mse_pred : str
        Reduction type for MSE prediction loss.
    kloss : int
        Number of clusters for cluster loss.
    klmbda : float
        Lambda value for cluster loss.
    bsize : int
        Size of the batch.
    noise : bool
        Whether to add Gaussian noise to the input.
    writer : Optional[SummaryWriter]
        TensorBoard writer for logging.
    global_step : int
        Global step counter for TensorBoard logging.

    Returns
    -------
    Tuple[float, float, float, float, float, float, int]
        Kullback-Leibler weight, train loss, K-means loss, KL loss,
        MSE loss, future loss, updated global step.
    """
    # toggle model to train mode
    model.train()
    train_loss = 0.0
    mse_loss = 0.0
    kullback_loss = 0.0
    kmeans_losses = 0.0
    fut_loss = 0.0
    loss = 0.0
    seq_len_half = int(seq_len / 2)

    for idx, data_item in enumerate(train_loader):
        data_item = Variable(data_item)
        data_item = data_item.permute(0, 2, 1)
        if use_gpu:
            data = data_item[:, :seq_len_half, :].type("torch.FloatTensor").cuda()
            fut = data_item[:, seq_len_half : seq_len_half + future_steps, :].type("torch.FloatTensor").cuda()
        else:
            data = data_item[:, :seq_len_half, :].type("torch.FloatTensor").to()
            fut = data_item[:, seq_len_half : seq_len_half + future_steps, :].type("torch.FloatTensor").to()

        if noise is True:
            data_gaussian = gaussian(data, True, seq_len_half)
        else:
            data_gaussian = data

        if future_decoder:
            data_tilde, future, latent, mu, logvar = model(data_gaussian)
            rec_loss = reconstruction_loss(data, data_tilde, mse_red)
            fut_rec_loss = future_reconstruction_loss(fut, future, mse_pred)
            kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
            kl_loss = kullback_leibler_loss(mu, logvar)
            kl_weight = kl_annealing(epoch, kl_start, annealtime, anneal_function)
            loss = rec_loss + fut_rec_loss + BETA * kl_weight * kl_loss + kl_weight * kmeans_loss
            fut_loss += fut_rec_loss.item()
        else:
            data_tilde, latent, mu, logvar = model(data_gaussian)
            rec_loss = reconstruction_loss(data, data_tilde, mse_red)
            kl_loss = kullback_leibler_loss(mu, logvar)
            kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
            kl_weight = kl_annealing(epoch, kl_start, annealtime, anneal_function)
            loss = rec_loss + BETA * kl_weight * kl_loss + kl_weight * kmeans_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TensorBoard logging every N batches
        if writer and idx % TENSORBOARD_LOG_FREQUENCY == 0:
            step = global_step + idx
            writer.add_scalar('batch/train_loss', loss.item(), step)
            writer.add_scalar('batch/mse_loss', rec_loss.item(), step)
            writer.add_scalar('batch/kl_loss', kl_loss.item(), step)
            writer.add_scalar('batch/kmeans_loss', kmeans_loss.item(), step)
            writer.add_scalar('batch/kl_weight', kl_weight, step)
            writer.add_scalar('batch/learning_rate', optimizer.param_groups[0]['lr'], step)

            if future_decoder:
                writer.add_scalar('batch/future_loss', fut_rec_loss.item(), step)

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5)

        train_loss += loss.item()
        mse_loss += rec_loss.item()
        kullback_loss += kl_loss.item()
        kmeans_losses += kmeans_loss.item()

        # if idx % 1000 == 0:
        #     print('Epoch: %d.  loss: %.4f' %(epoch, loss.item()))

    # be sure scheduler is called before optimizer in >1.1 pytorch
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(loss)
    else:
        scheduler.step()

    if future_decoder:
        logger.info(
            "Train loss: {:.3f}, MSE-Loss: {:.3f}, MSE-Future-Loss {:.3f}, KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}, weight: {:.2f}".format(
                train_loss / idx,
                mse_loss / idx,
                fut_loss / idx,
                BETA * kl_weight * kullback_loss / idx,
                kl_weight * kmeans_losses / idx,
                kl_weight,
            )
        )
    else:
        logger.info(
            "Train loss: {:.3f}, MSE-Loss: {:.3f}, KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}, weight: {:.2f}".format(
                train_loss / idx,
                mse_loss / idx,
                BETA * kl_weight * kullback_loss / idx,
                kl_weight * kmeans_losses / idx,
                kl_weight,
            )
        )

    return (
        kl_weight,
        train_loss / idx,
        kl_weight * kmeans_losses / idx,
        kullback_loss / idx,
        mse_loss / idx,
        fut_loss / idx,
        global_step + len(train_loader),
    )


def test(
    test_loader: Data.DataLoader,
    model: nn.Module,
    BETA: float,
    kl_weight: float,
    seq_len: int,
    mse_red: str,
    kloss: int,
    klmbda: float,
    future_decoder: bool,
    bsize: int,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
) -> Tuple[float, float, float]:
    """
    Evaluate the model on the test dataset.

    Parameters
    ----------
    test_loader : DataLoader
        DataLoader for the test dataset.
    model : nn.Module
        The trained model.
    BETA : float
        Beta value for the VAE loss.
    kl_weight : float
        Weighting factor for the KL divergence loss.
    seq_len : int
        Length of the sequence.
    mse_red : str
        Reduction method for the MSE loss.
    kloss : int
        Loss function for K-means clustering.
    klmbda : float
        Lambda value for K-means loss.
    future_decoder : bool
        Flag indicating whether to use a future decoder.
    bsize : int
        Batch size.
    writer : Optional[SummaryWriter]
        TensorBoard writer for logging.
    epoch : int
        Current epoch number.

    Returns
    -------
    Tuple[float, float, float]
        Tuple containing MSE loss per item, total test loss per item,
        and K-means loss weighted by the kl_weight.
    """
    # toggle model to inference mode
    model.eval()
    test_loss = 0.0
    mse_loss = 0.0
    kullback_loss = 0.0
    kmeans_losses = 0.0
    loss = 0.0
    seq_len_half = int(seq_len / 2)

    with torch.no_grad():
        for idx, data_item in enumerate(test_loader):
            # we're only going to infer, so no autograd at all required
            data_item = Variable(data_item)
            data_item = data_item.permute(0, 2, 1)
            if use_gpu:
                data = data_item[:, :seq_len_half, :].type("torch.FloatTensor").cuda()
            else:
                data = data_item[:, :seq_len_half, :].type("torch.FloatTensor").to()

            if future_decoder:
                recon_images, _, latent, mu, logvar = model(data)
                rec_loss = reconstruction_loss(data, recon_images, mse_red)
                kl_loss = kullback_leibler_loss(mu, logvar)
                kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
                loss = rec_loss + BETA * kl_weight * kl_loss + kl_weight * kmeans_loss

            else:
                recon_images, latent, mu, logvar = model(data)
                rec_loss = reconstruction_loss(data, recon_images, mse_red)
                kl_loss = kullback_leibler_loss(mu, logvar)
                kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
                loss = rec_loss + BETA * kl_weight * kl_loss + kl_weight * kmeans_loss

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5)

            test_loss += loss.item()
            mse_loss += rec_loss.item()
            kullback_loss += kl_loss.item()
            kmeans_losses += kmeans_loss.item()

    # TensorBoard logging for test metrics
    if writer:
        writer.add_scalar('epoch/test_loss', test_loss / idx, epoch)
        writer.add_scalar('epoch/test_mse', mse_loss / idx, epoch)
        writer.add_scalar('epoch/test_kl', BETA * kl_weight * kullback_loss / idx, epoch)
        writer.add_scalar('epoch/test_kmeans', kl_weight * kmeans_losses / idx, epoch)

    logger.info(
        "Test loss: {:.3f}, MSE-Loss: {:.3f}, KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}".format(
            test_loss / idx,
            mse_loss / idx,
            BETA * kl_weight * kullback_loss / idx,
            kl_weight * kmeans_losses / idx,
        )
    )
    return mse_loss / idx, test_loss / idx, kl_weight * kmeans_losses


@save_state(model=TrainModelFunctionSchema)
def train_model(
    config: dict,
    save_logs: bool = True,
) -> None:
    """
    Train Variational Autoencoder using the configuration file values.
    Fills in the values in the "train_model" key of the states.json file.
    Creates files at:
    - project_name/
        - model/
            - best_model/
                - snapshots/
                    - model_name_Project_epoch_0.pkl
                    - ...
                - model_name_Project.pkl
            - model_losses/
                - fut_losses_VAME.npy
                - kl_losses_VAME.npy
                - kmeans_losses_VAME.npy
                - mse_test_losses_VAME.npy
                - mse_train_losses_VAME.npy
                - test_losses_VAME.npy
                - train_losses_VAME.npy
                - weight_values_VAME.npy
            - pretrained_model/
        - logs/
            - tensorboard/
                - model_name/
                    - events.out.tfevents...


    Parameters
    ----------
    config : dict
        Configuration dictionary.
    save_logs : bool, optional
        Whether to save the logs. Default is True.

    Returns
    -------
    None
    """
    config = config
    writer = None
    try:
        tqdm_logger_stream = None
        if save_logs:
            tqdm_logger_stream = TqdmToLogger(logger)
            log_path = Path(config["project_path"]) / "logs" / "train_model.log"
            logger_config.add_file_handler(str(log_path))

        model_name = config["model_name"]
        pretrained_weights = config["pretrained_weights"]
        pretrained_model = config["pretrained_model"]
        fixed = config["egocentric_data"]

        logger.info("Train Variational Autoencoder - model name: %s \n" % model_name)
        if not os.path.exists(os.path.join(config["project_path"], "model", "best_model", "")):
            os.mkdir(os.path.join(config["project_path"], "model", "best_model", ""))
            os.mkdir(os.path.join(config["project_path"], "model", "best_model", "snapshots", ""))
            os.mkdir(os.path.join(config["project_path"], "model", "model_losses", ""))

        # TensorBoard setup
        if TENSORBOARD_ENABLED:
            tb_log_dir = os.path.join(
                config["project_path"],
                "logs",
                "tensorboard",
                f"{model_name}"
            )
            os.makedirs(tb_log_dir, exist_ok=True)
            writer = SummaryWriter(tb_log_dir)
            logger.info(f"TensorBoard logging enabled. Log directory: {tb_log_dir}")
            logger.info("To view logs, run: tensorboard --logdir=%s --port=6006" % os.path.join(config["project_path"], "logs", "tensorboard"))

        # make sure torch uses cuda for GPU computing
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            logger.info("Using CUDA")
            logger.info("GPU active: {}".format(torch.cuda.is_available()))
            logger.info("GPU used: {}".format(torch.cuda.get_device_name(0)))
        else:
            torch.device("cpu")
            logger.info("warning, a GPU was not found... proceeding with CPU (slow!) \n")
            # raise NotImplementedError('GPU Computing is required!')

        # HYPERPARAMETERS
        # General
        CUDA = use_gpu
        SEED = 19
        TRAIN_BATCH_SIZE = config["batch_size"]
        TEST_BATCH_SIZE = int(config["batch_size"] / 4)
        EPOCHS = config["max_epochs"]
        ZDIMS = config["zdims"]
        BETA = config["beta"]
        SNAPSHOT = config["model_snapshot"]
        LEARNING_RATE = config["learning_rate"]
        NUM_FEATURES = config["num_features"]
        if not fixed:
            NUM_FEATURES = NUM_FEATURES - 3
        TEMPORAL_WINDOW = config["time_window"] * 2
        FUTURE_DECODER = config["prediction_decoder"]
        FUTURE_STEPS = config["prediction_steps"]

        # RNN
        hidden_size_layer_1 = config["hidden_size_layer_1"]
        hidden_size_layer_2 = config["hidden_size_layer_2"]
        hidden_size_rec = config["hidden_size_rec"]
        hidden_size_pred = config["hidden_size_pred"]
        dropout_encoder = config["dropout_encoder"]
        dropout_rec = config["dropout_rec"]
        dropout_pred = config["dropout_pred"]
        noise = config["noise"]
        scheduler_step_size = config["scheduler_step_size"]
        softplus = config["softplus"]

        # Loss
        MSE_REC_REDUCTION = config["mse_reconstruction_reduction"]
        MSE_PRED_REDUCTION = config["mse_prediction_reduction"]
        KMEANS_LOSS = config["kmeans_loss"]
        KMEANS_LAMBDA = config["kmeans_lambda"]
        KL_START = config["kl_start"]
        ANNEALTIME = config["annealtime"]
        anneal_function = config["anneal_function"]
        optimizer_scheduler = config["scheduler"]

        BEST_LOSS = 999999
        convergence = 0
        logger.info(
            "Latent Dimensions: %d, Time window: %d, Batch Size: %d, Beta: %d, lr: %.4f\n"
            % (ZDIMS, config["time_window"], TRAIN_BATCH_SIZE, BETA, LEARNING_RATE)
        )

        # simple logging of diverse losses
        train_losses = []
        test_losses = []
        kmeans_losses = []
        kl_losses = []
        weight_values = []
        mse_losses = []
        fut_losses = []

        torch.manual_seed(SEED)
        RNN = RNN_VAE
        if CUDA:
            torch.cuda.manual_seed(SEED)
            model = RNN(
                TEMPORAL_WINDOW,
                ZDIMS,
                NUM_FEATURES,
                FUTURE_DECODER,
                FUTURE_STEPS,
                hidden_size_layer_1,
                hidden_size_layer_2,
                hidden_size_rec,
                hidden_size_pred,
                dropout_encoder,
                dropout_rec,
                dropout_pred,
                softplus,
            ).cuda()
        else:  # cpu support ...
            torch.cuda.manual_seed(SEED)
            model = RNN(
                TEMPORAL_WINDOW,
                ZDIMS,
                NUM_FEATURES,
                FUTURE_DECODER,
                FUTURE_STEPS,
                hidden_size_layer_1,
                hidden_size_layer_2,
                hidden_size_rec,
                hidden_size_pred,
                dropout_encoder,
                dropout_rec,
                dropout_pred,
                softplus,
            ).to()

        # Log model graph to TensorBoard
        if writer:
            try:
                dummy_input = torch.randn(1, TEMPORAL_WINDOW // 2, NUM_FEATURES)
                if CUDA:
                    dummy_input = dummy_input.cuda()
                writer.add_graph(model, dummy_input)
                logger.info("Model graph logged to TensorBoard")
            except Exception as e:
                logger.warning(f"Could not log model graph to TensorBoard: {e}")

        if pretrained_weights:
            try:
                logger.info(
                    "Loading pretrained weights from model: %s\n"
                    % os.path.join(
                        config["project_path"],
                        "model",
                        "best_model",
                        pretrained_model + "_" + config["project_name"] + ".pkl",
                    )
                )
                model.load_state_dict(
                    torch.load(
                        os.path.join(
                            config["project_path"],
                            "model",
                            "best_model",
                            pretrained_model + "_" + config["project_name"] + ".pkl",
                        )
                    )
                )
                KL_START = 0
                ANNEALTIME = 1
            except Exception:
                logger.info(
                    "No file found at %s\n"
                    % os.path.join(
                        config["project_path"],
                        "model",
                        "best_model",
                        pretrained_model + "_" + config["project_name"] + ".pkl",
                    )
                )
                try:
                    logger.info("Loading pretrained weights from %s\n" % pretrained_model)
                    model.load_state_dict(torch.load(pretrained_model))
                    KL_START = 0
                    ANNEALTIME = 1
                except Exception:
                    logger.error("Could not load pretrained model. Check file path in config.yaml.")

        """ DATASET """
        trainset = SEQUENCE_DATASET(
            os.path.join(config["project_path"], "data", "train", ""),
            data="train_seq.npy",
            train=True,
            temporal_window=TEMPORAL_WINDOW,
        )
        testset = SEQUENCE_DATASET(
            os.path.join(config["project_path"], "data", "train", ""),
            data="test_seq.npy",
            train=False,
            temporal_window=TEMPORAL_WINDOW,
        )

        train_loader = Data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

        if optimizer_scheduler:
            logger.info(
                "Scheduler step size: %d, Scheduler gamma: %.2f\n" % (scheduler_step_size, config["scheduler_gamma"])
            )
            # Thanks to @alexcwsmith for the optimized scheduler contribution
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor=config["scheduler_gamma"],
                patience=config["scheduler_step_size"],
                threshold=1e-3,
                threshold_mode="rel",
            )
        else:
            scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=1, last_epoch=-1)

        global_step = 0
        logger.info("Start training... ")
        for epoch in tqdm(
            range(1, EPOCHS),
            desc="Training Model",
            unit="epoch",
            file=tqdm_logger_stream,
        ):
            # print("Epoch: %d" %epoch)
            weight, train_loss, km_loss, kl_loss, mse_loss, fut_loss, global_step = train(
                train_loader,
                epoch,
                model,
                optimizer,
                anneal_function,
                BETA,
                KL_START,
                ANNEALTIME,
                TEMPORAL_WINDOW,
                FUTURE_DECODER,
                FUTURE_STEPS,
                scheduler,
                MSE_REC_REDUCTION,
                MSE_PRED_REDUCTION,
                KMEANS_LOSS,
                KMEANS_LAMBDA,
                TRAIN_BATCH_SIZE,
                noise,
                writer,
                global_step,
            )
            current_loss, test_loss, test_list = test(
                test_loader,
                model,
                BETA,
                weight,
                TEMPORAL_WINDOW,
                MSE_REC_REDUCTION,
                KMEANS_LOSS,
                KMEANS_LAMBDA,
                FUTURE_DECODER,
                TEST_BATCH_SIZE,
                writer,
                epoch,
            )

            # TensorBoard epoch-level logging
            if writer:
                writer.add_scalar('epoch/train_loss', train_loss, epoch)
                writer.add_scalar('epoch/train_mse', mse_loss, epoch)
                writer.add_scalar('epoch/train_kl', kl_loss, epoch)
                writer.add_scalar('epoch/train_kmeans', km_loss, epoch)
                writer.add_scalar('epoch/kl_weight', weight, epoch)
                writer.add_scalar('epoch/learning_rate', optimizer.param_groups[0]['lr'], epoch)

                if FUTURE_DECODER:
                    writer.add_scalar('epoch/train_future', fut_loss, epoch)

                # Log parameter histograms (optional)
                if TENSORBOARD_LOG_HISTOGRAMS:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(f'parameters/{name}', param, epoch)
                            writer.add_histogram(f'gradients/{name}', param.grad, epoch)

            # logging losses
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            kmeans_losses.append(km_loss)
            kl_losses.append(kl_loss)
            weight_values.append(weight)
            mse_losses.append(mse_loss)
            fut_losses.append(fut_loss)

            # save best model
            if weight > 0.99 and current_loss <= BEST_LOSS:
                BEST_LOSS = current_loss
                logger.info("Saving model!")
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        config["project_path"],
                        "model",
                        "best_model",
                        model_name + "_" + config["project_name"] + ".pkl",
                    ),
                )
                convergence = 0
            else:
                convergence += 1

            if epoch % SNAPSHOT == 0:
                logger.info("Saving model snapshot!\n")
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        config["project_path"],
                        "model",
                        "best_model",
                        "snapshots",
                        model_name + "_" + config["project_name"] + "_epoch_" + str(epoch) + ".pkl",
                    ),
                )

            if convergence > config["model_convergence"]:
                logger.info("Finished training...")
                logger.info(
                    "Model converged. Please check your model with vame.evaluate_model(). \n"
                    "You can also re-run vame.trainmodel() to further improve your model. \n"
                    'Make sure to set _pretrained_weights_ in your config.yaml to "true" \n'
                    "and plug your current model name into _pretrained_model_. \n"
                    'Hint: Set "model_convergence" in your config.yaml to a higher value. \n'
                    "\n"
                    "Next: \n"
                    "Use vame.segment_session() to identify behavioral motifs in your dataset!"
                )
                break

            # save logged losses
            np.save(
                os.path.join(
                    config["project_path"],
                    "model",
                    "model_losses",
                    "train_losses_" + model_name,
                ),
                train_losses,
            )
            np.save(
                os.path.join(
                    config["project_path"],
                    "model",
                    "model_losses",
                    "test_losses_" + model_name,
                ),
                test_losses,
            )
            np.save(
                os.path.join(
                    config["project_path"],
                    "model",
                    "model_losses",
                    "kmeans_losses_" + model_name,
                ),
                kmeans_losses,
            )
            np.save(
                os.path.join(
                    config["project_path"],
                    "model",
                    "model_losses",
                    "kl_losses_" + model_name,
                ),
                kl_losses,
            )
            np.save(
                os.path.join(
                    config["project_path"],
                    "model",
                    "model_losses",
                    "weight_values_" + model_name,
                ),
                weight_values,
            )
            np.save(
                os.path.join(
                    config["project_path"],
                    "model",
                    "model_losses",
                    "mse_train_losses_" + model_name,
                ),
                mse_losses,
            )
            np.save(
                os.path.join(
                    config["project_path"],
                    "model",
                    "model_losses",
                    "mse_test_losses_" + model_name,
                ),
                current_loss,
            )
            np.save(
                os.path.join(
                    config["project_path"],
                    "model",
                    "model_losses",
                    "fut_losses_" + model_name,
                ),
                fut_losses,
            )
            logger.info("\n")

        if convergence < config["model_convergence"]:
            logger.info("Finished training...")
            logger.info(
                "Model seems to have not reached convergence. You may want to check your model \n"
                "with vame.evaluate_model(). If your satisfied you can continue. \n"
                "Use vame.segment_session() to identify behavioral motifs! \n"
                "OPTIONAL: You can re-run vame.train_model() to improve performance."
            )

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise e
    finally:
        if writer:
            writer.close()
            logger.info("TensorBoard writer closed")
        logger_config.remove_file_handler()
