use crate::data::MnistBatcher;
use crate::display_tensor;
use crate::training::TrainingConfig;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::vision::MnistItem;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::record::Recorder;

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MnistItem) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    let label = item.label;
    let batcher = MnistBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let (ae_out, cls_out) = model.forward(batch.images.clone());
    let predicted = cls_out.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
    println!("Autoencoder Output:");
    display_tensor(ae_out.squeeze(0));
}
