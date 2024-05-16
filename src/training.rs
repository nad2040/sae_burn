use burn::prelude::*;
use burn::train::metric::*;
use derive_new::new;

#[derive(new)]
pub struct SAEOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: (Tensor<B, 3>, Tensor<B, 2>),
    pub targets: (Tensor<B, 3>, Tensor<B, 1, Int>),
}

impl<B: Backend> Adaptor<AccuracyInput<B>> for SAEOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        AccuracyInput::new(self.output.clone().1, self.targets.clone().1)
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for SAEOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

use crate::data::MnistBatch;
use crate::model::Model;
use burn::nn::loss::{CrossEntropyLoss, MseLoss, Reduction};
use burn::train::{TrainOutput, TrainStep, ValidStep};

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> SAEOutput<B> {
        let (ae_out, cls_out) = self.forward(images.clone());
        let loss = MseLoss::new().forward(ae_out.clone(), images.clone(), Reduction::Mean)
            + CrossEntropyLoss::new(None, &cls_out.device())
                .forward(cls_out.clone(), targets.clone());

        SAEOutput::new(loss, (ae_out, cls_out), (images, targets))
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, SAEOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<SAEOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, SAEOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> SAEOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

use crate::data::MnistBatcher;
use crate::model::ModelConfig;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::vision::MnistDataset;
use burn::optim::AdamConfig;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::LearnerBuilder;

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 15)]
    pub num_epochs: usize,
    #[config(default = 128)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-2)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = MnistBatcher::<B>::new(device.clone());
    let batcher_valid = MnistBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
