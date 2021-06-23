use sttt::util::lower_process_priority;

use sttt_zero::selfplay::{MCTSGeneratorSettings, MoveSelector, Settings};

fn main() {
    lower_process_priority();

    let settings = Settings {
        position_count: 2_100_000,
        output_path: "../data/esat2/all_data.csv".to_owned(),

        move_selector: MoveSelector {
            inf_temp_move_count: 20
        },

        generator: MCTSGeneratorSettings {
            thread_count: 32,

            iterations: 100_000,
            exploration_weight: 1.0,
        },
    };
    settings.run();
}
