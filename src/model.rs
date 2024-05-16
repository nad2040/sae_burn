use burn::{
    nn::{Dropout, DropoutConfig, LeakyRelu, LeakyReluConfig, Linear, LinearConfig, Relu},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    enc1: Linear<B>,
    enc2: Linear<B>,
    enc3: Linear<B>,
    bottleneck: Linear<B>,
    dec1: Linear<B>,
    dec2: Linear<B>,
    dec3: Linear<B>,
    ae_out: Linear<B>,
    cls1: Linear<B>,
    cls2: Linear<B>,
    cls_out: Linear<B>,
    lrelu: LeakyRelu<B>,
    relu: Relu,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    h1: usize,
    h2: usize,
    h3: usize,
    k1: usize,
    k2: usize,
    #[config(default = "2")]
    bottleneck: usize,
    #[config(default = "0.3")]
    lrelu: f64,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            enc1: LinearConfig::new(28 * 28, self.h1).init(device),
            enc2: LinearConfig::new(self.h1, self.h2).init(device),
            enc3: LinearConfig::new(self.h2, self.h3).init(device),
            bottleneck: LinearConfig::new(self.h3, self.bottleneck).init(device),
            dec1: LinearConfig::new(self.bottleneck, self.h3).init(device),
            dec2: LinearConfig::new(self.h3, self.h2).init(device),
            dec3: LinearConfig::new(self.h2, self.h1).init(device),
            ae_out: LinearConfig::new(self.h1, 28 * 28).init(device),
            cls1: LinearConfig::new(self.bottleneck, self.k1).init(device),
            cls2: LinearConfig::new(self.k1, self.k2).init(device),
            cls_out: LinearConfig::new(self.k2, 10).init(device),
            lrelu: LeakyReluConfig::new().init(),
            relu: Relu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [batch_size, height, width] = images.dims();

        let x = images.reshape([batch_size, height * width]);

        let x = self.dropout.forward(x);
        let x = self.enc1.forward(x);
        let x = self.lrelu.forward(x);
        let x = self.dropout.forward(x);
        let x = self.enc2.forward(x);
        let x = self.lrelu.forward(x);
        let x = self.enc3.forward(x);
        let x = self.lrelu.forward(x);

        let x = self.bottleneck.forward(x);
        let bottleneck = self.lrelu.forward(x);

        let ae = self.dec1.forward(bottleneck.clone());
        let ae = self.lrelu.forward(ae);
        let ae = self.dec2.forward(ae);
        let ae = self.lrelu.forward(ae);
        let ae = self.dropout.forward(ae);
        let ae = self.dec3.forward(ae);
        let ae = self.lrelu.forward(ae);
        let ae = self.dropout.forward(ae);
        let ae = self.ae_out.forward(ae);
        let ae = self.relu.forward(ae);
        let ae_out = ae.reshape([batch_size, height, width]);

        let cls = self.cls1.forward(bottleneck);
        let cls = self.lrelu.forward(cls);
        let cls = self.dropout.forward(cls);
        let cls = self.cls2.forward(cls);
        let cls = self.lrelu.forward(cls);
        let cls = self.cls_out.forward(cls);
        let cls_out = self.lrelu.forward(cls);

        (ae_out, cls_out)
    }
}
