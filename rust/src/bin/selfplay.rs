use sttt::util::lower_process_priority;
use tch::Device;

use sttt_zero::selfplay::{MoveSelector, Settings};
use sttt_zero::selfplay::generate_zero::settings_torch::GoogleTorchSettings;
use sttt_zero::selfplay::generate_zero::ZeroGeneratorSettings;

fn main() {
    lower_process_priority();

    let settings = Settings {
        position_count: 100_000,
        output_path: "../data/esat2/data_from_moderate_5.csv".to_owned(),

        move_selector: MoveSelector {
            inf_temp_move_count: 20
        },

        generator: ZeroGeneratorSettings {
            batch_size: 1000,
            network: GoogleTorchSettings {
                path: "../data/esat2/modest/model_5_epochs.pt".to_owned(),
                devices: vec![Device::Cuda(0), Device::Cuda(1)],
                threads_per_device: 2,
            },
            iterations: 5_000,
            exploration_weight: 1.0,
        },
    };
    settings.run();
}
