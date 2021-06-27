use std::time::Instant;

use sttt::board::Board;

use sttt_zero::network::google_onnx::GoogleOnnxNetwork;
use sttt_zero::network::Network;
use ta::indicators::ExponentialMovingAverage;
use ta::Next;
use sttt_zero::network::google_torch::GoogleTorchNetwork;
use tch::Device;
use sttt_zero::network::google_tract::GoogleTractNetwork;

// (everything with batch size 1000)
// ONNX: <10k, seems to be using cpu
// TORCH CPU:  still broken, complaining about None
// TORCH Cuda: same
// Tract:

fn main() {
    let mut network = GoogleTractNetwork::load("../data/esat2/modest/model_5_epochs.onnx");
    let batch_size = 100;
    let batch = vec![Board::new(); batch_size];
    let mut throughput_ema = ExponentialMovingAverage::new(100).unwrap();

    loop {
        let start = Instant::now();
        network.evaluate_batch(&batch);

        let delta = Instant::now() - start;
        let throughput = (batch_size as f64) / delta.as_secs_f64();

        println!("Throughput: {:.02} boards/s", throughput_ema.next(throughput));
    }
}