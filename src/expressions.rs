#![allow(clippy::unused_unit)]
use crate::pairing::*;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct PairingKwargs {
    method: String,
}

fn pair_struct(_input_field: &[Field]) -> PolarsResult<Field> {
    let left = Field::new("Left".into(), DataType::UInt64);
    let right = Field::new("Right".into(), DataType::UInt64);
    Ok(Field::new(
        "Pair".into(),
        DataType::Struct(vec![left, right]),
    ))
}

#[polars_expr(output_type=UInt64)]
fn pair(inputs: &[Series], kwargs: PairingKwargs) -> PolarsResult<Series> {
    // Match the method string to the pairing method, otherwise return an error
    let method = match kwargs.method.to_ascii_lowercase().as_str() {
        "hagen" => "hagen",
        "szudzik" => "szudzik",
        "cantor" => "cantor",
        _ => {
            polars_bail!(InvalidOperation: "Invalid pairing method: {}", kwargs.method);
        }
    };
    // Ensure both inputs are of integer type, otherwise return an error
    if !inputs[0].dtype().is_integer() || !inputs[1].dtype().is_integer() {
        polars_bail!(InvalidOperation: "The pairing function can only be used on integers.");
    }
    // Cast the first and second input series to UInt64, unwrapping the result directly
    let left_series = inputs[0].cast(&DataType::UInt64).unwrap();
    let right_series = inputs[1].cast(&DataType::UInt64).unwrap();
    // Convert the series to their unsigned 64-bit integer chunked arrays
    let left = left_series.u64()?;
    let right = right_series.u64()?;
    // Apply the pairing function element-wise to left and right
    // If the method is "szudzik", use the Szudzik pairing function,
    // otherwise use the Cantor pairing function
    let out: UInt64Chunked = arity::binary_elementwise_values(
        left,
        right,
        match method {
            "hagen" => compute_hagen_pair,
            "szudzik" => compute_szudzik_pair,
            "cantor" => compute_cantor_pair,
            _ => unreachable!(),
        },
    );
    // Return the result as a Series wrapped in Ok
    Ok(out.into_series())
}

#[polars_expr(output_type_func=pair_struct)]
fn unpair(inputs: &[Series], kwargs: PairingKwargs) -> PolarsResult<Series> {
    // Match the method string to the pairing method, otherwise return an error
    let method = match kwargs.method.to_ascii_lowercase().as_str() {
        "hagen" => "hagen",
        "szudzik" => "szudzik",
        "cantor" => "cantor",
        _ => {
            polars_bail!(InvalidOperation: "Invalid pairing method: {}", kwargs.method);
        }
    };
    // Ensure the input is an integer, otherwise return an error
    if !inputs[0].dtype().is_integer() {
        polars_bail!(InvalidOperation: "The unpairing function can only be used on integers.");
    }
    // Cast the input series to UInt64, unwrapping the result directly
    let paired_series = inputs[0].cast(&DataType::UInt64).unwrap();
    // Convert the series to their 64-bit integer chunked arrays
    let paired = paired_series.u64()?;
    // Apply the unpairing function element-wise to paired
    // If the method is "szudzik", use the Szudzik unpairing function,
    // otherwise use the Cantor unpairing function
    let unpaired: Vec<Option<(u64, u64)>> = paired
        .iter()
        .map(|n| match method {
            "hagen" => n.and_then(compute_hagen_unpair),
            "szudzik" => n.and_then(compute_szudzik_unpair),
            "cantor" => n.and_then(compute_cantor_unpair),
            _ => unreachable!(),
        })
        .collect();
    // Create an iterator that takes the left value of the pair if it exists or 0 otherwise
    let left: Vec<Option<u64>> = unpaired.iter().map(|p| p.map(|(l, _)| l)).collect();
    let right: Vec<Option<u64>> = unpaired.iter().map(|p| p.map(|(_, r)| r)).collect();
    // Create a StructChunked from the left and right vectors
    let left_series = UInt64Chunked::from_iter(left)
        .with_name("Left".into())
        .into_series();
    let right_series = UInt64Chunked::from_iter(right)
        .with_name("Right".into())
        .into_series();
    let out = StructChunked::from_series(
        "Pair".into(),
        left_series.len(),
        [&left_series, &right_series].into_iter(),
    )?;
    // Return the result as a Series wrapped in Ok
    Ok(out.into_series())
}
