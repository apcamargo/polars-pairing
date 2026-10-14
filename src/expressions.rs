#![allow(clippy::unused_unit)]
use pairing_functions::{
    compute_cantor_pair, compute_cantor_unpair, compute_hagen_pair, compute_hagen_unpair,
    compute_peter_pair, compute_peter_unpair, compute_rosenberg_strong_pair,
    compute_rosenberg_strong_unpair, compute_szudzik_pair, compute_szudzik_unpair,
    compute_thotakura_pair, compute_thotakura_unpair,
};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct PairingKwargs {
    method: String,
}

#[derive(Copy, Clone)]
enum Method {
    Hagen,
    Szudzik,
    Cantor,
    Peter,
    RosenbergStrong,
    Thotakura,
}

impl Method {
    /// Resolve a method name (case-insensitive) to a `Method`, failing with
    /// the user-supplied name if it is not recognized.
    fn parse(method: &str, operation: &str) -> PolarsResult<Self> {
        match method.to_ascii_lowercase().as_str() {
            "hagen" => Ok(Self::Hagen),
            "szudzik" => Ok(Self::Szudzik),
            "cantor" => Ok(Self::Cantor),
            "peter" => Ok(Self::Peter),
            "rosenberg_strong" => Ok(Self::RosenbergStrong),
            "thotakura" => Ok(Self::Thotakura),
            _ => polars_bail!(InvalidOperation: "Invalid {} method: {}", operation, method),
        }
    }

    /// Canonical (lowercase) name, for use in error messages.
    fn name(self) -> &'static str {
        match self {
            Self::Hagen => "hagen",
            Self::Szudzik => "szudzik",
            Self::Cantor => "cantor",
            Self::Peter => "peter",
            Self::RosenbergStrong => "rosenberg_strong",
            Self::Thotakura => "thotakura",
        }
    }
}

/// Pair two values with the given method, mapping overflow into a
/// `PolarsError` with method and inputs for context.
fn pair_one(method: Method, l: u64, r: u64) -> PolarsResult<u64> {
    let result = match method {
        Method::Hagen => compute_hagen_pair(l, r),
        Method::Szudzik => compute_szudzik_pair(l, r),
        Method::Cantor => compute_cantor_pair(l, r),
        Method::Peter => compute_peter_pair(l, r),
        Method::RosenbergStrong => compute_rosenberg_strong_pair(l, r),
        Method::Thotakura => compute_thotakura_pair(l, r),
    };
    result.map_err(|_| {
        polars_err!(
            InvalidOperation: "pairing overflow for method '{}' with inputs ({}, {})",
            method.name(),
            l,
            r
        )
    })
}

fn unpair_one(method: Method, p: u64) -> PolarsResult<(u64, u64)> {
    let result = match method {
        Method::Hagen => compute_hagen_unpair(p),
        Method::Szudzik => compute_szudzik_unpair(p),
        Method::Cantor => compute_cantor_unpair(p),
        Method::Peter => compute_peter_unpair(p),
        Method::RosenbergStrong => compute_rosenberg_strong_unpair(p),
        Method::Thotakura => compute_thotakura_unpair(p),
    };
    result.map_err(|_| {
        polars_err!(
            InvalidOperation: "unpairing failure for method '{}' with input ({})",
            method.name(),
            p
        )
    })
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
    // Resolve the method up front so invalid names fail even on empty input
    let method = Method::parse(&kwargs.method, "pairing")?;
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
    // Nulls propagate without invoking the function; overflow raises an error
    let out: UInt64Chunked = arity::try_binary_elementwise(left, right, |l, r| match (l, r) {
        (Some(l), Some(r)) => pair_one(method, l, r).map(Some),
        _ => Ok(None),
    })?;
    // Return the result as a Series wrapped in Ok
    Ok(out.into_series())
}

#[polars_expr(output_type_func=pair_struct)]
fn unpair(inputs: &[Series], kwargs: PairingKwargs) -> PolarsResult<Series> {
    // Resolve the method up front so invalid names fail even on empty input
    let method = Method::parse(&kwargs.method, "unpairing")?;
    // Ensure the input is an integer, otherwise return an error
    if !inputs[0].dtype().is_integer() {
        polars_bail!(InvalidOperation: "The unpairing function can only be used on integers.");
    }
    // Cast the input series to UInt64, unwrapping the result directly
    let paired_series = inputs[0].cast(&DataType::UInt64).unwrap();
    // Convert the series to their 64-bit integer chunked arrays
    let paired = paired_series.u64()?;
    // Apply the unpairing function element-wise to paired
    let unpaired: Vec<Option<(u64, u64)>> = paired
        .iter()
        .map(|n| n.map(|p| unpair_one(method, p)).transpose())
        .collect::<PolarsResult<_>>()?;
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
