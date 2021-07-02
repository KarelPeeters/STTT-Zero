use sttt::util::lower_process_priority;

use sttt_zero::selfplay::{MoveSelector, Settings};
use sttt_zero::selfplay::generate_mcts::MCTSGeneratorSettings;
use sttt_zero::zero::ZeroSettings;

fn main() {
    lower_process_priority();

    let settings = Settings {
        game_count: 2_000_000,
        output_path: "../data/esat3/games.csv".to_owned(),

        move_selector: MoveSelector {
            inf_temp_move_count: 20
        },

        generator: MCTSGeneratorSettings {
            thread_count: 16,
            iterations: 200_000,
            exploration_weight: 2.0,
        },
    };
    settings.run();
}
