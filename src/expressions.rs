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

/// Index of the first row where `orig` holds a value that the non-strict cast
/// to `cast` turned into null. Genuine nulls in `orig` are skipped. The
/// null-count comparison keeps the common (no failure) path cheap.
fn first_unconvertible(orig: &Series, cast: &Series) -> Option<usize> {
    if cast.null_count() == orig.null_count() {
        return None;
    }
    orig.iter()
        .zip(cast.iter())
        .position(|(o, c)| !o.is_null() && c.is_null())
}

/// Integer inputs are paired; an all-null (Null dtype) input such as
/// `pl.lit(None)` is accepted too and simply yields nulls.
fn is_accepted(dtype: &DataType) -> bool {
    dtype.is_integer() || dtype.is_null()
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
    // Ensure both inputs are integers (or all-null), otherwise return an error
    if !is_accepted(inputs[0].dtype()) || !is_accepted(inputs[1].dtype()) {
        polars_bail!(InvalidOperation: "The pairing function can only be used on integers.");
    }
    // A length-1 input (scalar or literal) is broadcast against the other
    // side below; any other length mismatch is an error.
    let len = match (inputs[0].len(), inputs[1].len()) {
        (1, r) => r,
        (l, r) if l == r || r == 1 => l,
        (l, r) => polars_bail!(
            ShapeMismatch: "cannot evaluate two Series of different lengths ({} and {})", l, r
        ),
    };
    // Non-strict cast: negatives and values beyond u64::MAX become null, and
    // are reported as errors below. Genuine nulls stay null.
    let left_series = inputs[0].cast(&DataType::UInt64)?;
    let right_series = inputs[1].cast(&DataType::UInt64)?;
    let bad_row = [
        first_unconvertible(&inputs[0], &left_series),
        first_unconvertible(&inputs[1], &right_series),
    ]
    .into_iter()
    .flatten()
    .min();
    // With no output rows, an out-of-range broadcast value is never paired,
    // so there is nothing to report.
    if let Some(row) = bad_row.filter(|_| len > 0) {
        // A length-1 input is broadcast, so its only value sits at index 0.
        fn at(s: &Series, row: usize) -> PolarsResult<AnyValue<'_>> {
            s.get(if s.len() == 1 { 0 } else { row })
        }
        polars_bail!(
            InvalidOperation: "pairing input out of range for method '{}' with inputs ({}, {})",
            method.name(),
            at(&inputs[0], row)?,
            at(&inputs[1], row)?
        );
    }
    // Convert the series to their unsigned 64-bit integer chunked arrays
    let left = left_series.u64()?;
    let right = right_series.u64()?;
    // Nulls propagate without invoking the function; overflow raises an error.
    // A length-1 side is applied as a scalar rather than materialized.
    let out: UInt64Chunked =
        arity::broadcast_try_binary_elementwise(left, right, |l, r| match (l, r) {
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
    // Ensure the input is an integer (or all-null), otherwise return an error
    if !is_accepted(inputs[0].dtype()) {
        polars_bail!(InvalidOperation: "The unpairing function can only be used on integers.");
    }
    // Non-strict cast: negatives and values beyond u64::MAX become null, and
    // are reported as errors. Genuine nulls stay null.
    let paired_series = inputs[0].cast(&DataType::UInt64)?;
    if let Some(row) = first_unconvertible(&inputs[0], &paired_series) {
        polars_bail!(
            InvalidOperation: "unpairing input out of range for method '{}' with input ({})",
            method.name(),
            inputs[0].get(row)?
        );
    }
    // Convert the series to their 64-bit integer chunked arrays
    let paired = paired_series.u64()?;
    let mut left = PrimitiveChunkedBuilder::<UInt64Type>::new("Left".into(), paired.len());
    let mut right = PrimitiveChunkedBuilder::<UInt64Type>::new("Right".into(), paired.len());
    for n in paired.iter() {
        match n {
            None => {
                left.append_null();
                right.append_null();
            }
            Some(p) => {
                let (l, r) = unpair_one(method, p)?;
                left.append_value(l);
                right.append_value(r);
            }
        }
    }
    // Create a StructChunked from the left and right columns
    let left_series = left.finish().into_series();
    let right_series = right.finish().into_series();
    let out = StructChunked::from_series(
        "Pair".into(),
        paired.len(),
        [&left_series, &right_series].into_iter(),
    )?;
    // Return the result as a Series wrapped in Ok
    Ok(out.into_series())
}
