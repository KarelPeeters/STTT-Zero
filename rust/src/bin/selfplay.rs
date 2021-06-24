#![allow(unused_imports)]

use sttt::util::lower_process_priority;

use sttt_zero::network::google_onnx::GoogleOnnxNetwork;
use sttt_zero::selfplay::{MoveSelector, Settings};
use sttt_zero::selfplay::generate_mcts::MCTSGeneratorSettings;
use sttt_zero::selfplay::generate_zero::settings_onnx::GoogleOnnxSettings;
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
            network: GoogleOnnxSettings {
                path: "../data/esat2/moderate/model_5_epochs.onnx".to_owned(),
                num_threads: 4,
            },
            iterations: 5_000,
            exploration_weight: 1.0,
        },
    };
    settings.run();
}
