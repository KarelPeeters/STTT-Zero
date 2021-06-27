use sttt::board::Board;
use tract_onnx::prelude::*;

use crate::network::{collect_google_output, encode_google_input, Network, NetworkEvaluation};

// type Model = SimplePlan<InferenceFact, Box<dyn InferenceOp>, tract_onnx::prelude::Graph<InferenceFact, Box<dyn InferenceOp>>>;
type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct GoogleTractNetwork {
    model: Model,
}

impl GoogleTractNetwork {
    pub fn load(path: &str) -> GoogleTractNetwork {
        let batch = Symbol::new('B');
        let input_shape = [batch.to_dim(), 5usize.into(), 9usize.into(), 9usize.into()];

        let model = tract_onnx::onnx()
            .model_for_path(path).unwrap()
            .with_input_fact(0, InferenceFact::dt_shape(DatumType::F32, &input_shape)).unwrap()
            .with_output_fact(0, InferenceFact::default()).unwrap()
            .with_output_fact(1, InferenceFact::default()).unwrap()
            .into_optimized().unwrap()
            .into_runnable().unwrap();

        GoogleTractNetwork { model }
    }
}

impl Network for GoogleTractNetwork {
    fn evaluate_batch(&mut self, boards: &[Board]) -> Vec<NetworkEvaluation> {
        let batch_size = boards.len();

        let input = encode_google_input(boards);
        let input = tract_ndarray::Array::from_shape_vec((batch_size, 5, 9, 9), input)
            .unwrap();

        let output = self.model.run(tvec!(input.into_tensor())).unwrap();
        assert_eq!(output.len(), 2);
        let value = &output[0];
        let policy = &output[1];
        assert_eq!(value.shape(), &[batch_size, 1]);
        assert_eq!(policy.shape(), &[batch_size, 81]);

        collect_google_output(boards, value.as_slice().unwrap(), policy.as_slice().unwrap())
    }
}
