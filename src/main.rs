mod data;
mod model;

mod inference;
mod training;

use crate::model::ModelConfig;
use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::data::dataloader::Dataset;
use burn::optim::AdamConfig;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub fn display_tensor<B: Backend>(t: Tensor<B, 2>) {
    let gradient = " .:-=+*#%@".as_bytes();
    let t = t
        .clone()
        .div(t.clone().max().unsqueeze())
        .mul_scalar(9.9)
        .int();
    for x in t.iter_dim(0) {
        for y in x.iter_dim(1) {
            let idx: usize = y.into_scalar().to_string().parse::<usize>().unwrap();
            print!("{}", gradient[idx] as char);
        }
        println!("");
    }
}

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let mut artifact_dir = project_root::get_project_root().unwrap_or(".".into());
    artifact_dir.push("training");
    let artifact_dir = artifact_dir.to_str().expect("Error: no root dir found");

    let train = false;
    if train {
        crate::training::train::<MyAutodiffBackend>(
            artifact_dir,
            crate::training::TrainingConfig::new(
                ModelConfig::new(150, 80, 35, 200, 20),
                AdamConfig::new(),
            ),
            device,
        );
    } else {
        crate::inference::infer::<MyBackend>(
            artifact_dir,
            device,
            burn::data::dataset::vision::MnistDataset::test()
                .get(42)
                .unwrap(),
        );
    }
}
