use std::convert::TryFrom;

use fearless_simd::{Level, dispatch, prelude::*};
use mlua::{AnyUserData, Error, Lua, Result, Table, Value, Variadic};

use crate::storage::{Storage, value_to_f32};

#[derive(Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
}

#[derive(Clone, Copy)]
enum UnaryOp {
    Relu,
    Scale(f32),
    Pow(f32),
}

pub(crate) fn register(lua: &Lua, exports: &Table, level: Level) -> Result<()> {
    let add_level = level;
    exports.set(
        "add_into",
        lua.create_function(move |_, args: Variadic<Value>| {
            binary_into(add_level, args, BinaryOp::Add)
        })?,
    )?;
    let sub_level = level;
    exports.set(
        "sub_into",
        lua.create_function(move |_, args: Variadic<Value>| {
            binary_into(sub_level, args, BinaryOp::Sub)
        })?,
    )?;
    let mul_level = level;
    exports.set(
        "mul_into",
        lua.create_function(move |_, args: Variadic<Value>| {
            binary_into(mul_level, args, BinaryOp::Mul)
        })?,
    )?;

    let relu_level = level;
    exports.set(
        "relu_into",
        lua.create_function(move |_, args: Variadic<Value>| {
            unary_into(relu_level, args, UnaryOp::Relu)
        })?,
    )?;
    let scale_level = level;
    exports.set(
        "scale_into",
        lua.create_function(move |_, args: Variadic<Value>| {
            let (input, scale, out) = parse_scalar_unary(args, "scale_into")?;
            unary_into_values(scale_level, input, out, UnaryOp::Scale(scale))
        })?,
    )?;
    let pow_level = level;
    exports.set(
        "pow_into",
        lua.create_function(move |_, args: Variadic<Value>| {
            let (input, exponent, out) = parse_scalar_unary(args, "pow_into")?;
            unary_into_values(pow_level, input, out, UnaryOp::Pow(exponent))
        })?,
    )?;

    let sum_level = level;
    exports.set(
        "sum",
        lua.create_function(move |_, args: Variadic<Value>| {
            let input = parse_one_storage(args, "sum")?;
            let values = snapshot_storage(&input, "sum input")?;
            Ok(run_sum(sum_level, &values))
        })?,
    )?;
    let mean_level = level;
    exports.set(
        "mean",
        lua.create_function(move |_, args: Variadic<Value>| {
            let input = parse_one_storage(args, "mean")?;
            let values = snapshot_storage(&input, "mean input")?;
            if values.is_empty() {
                return Err(Error::runtime("mean is undefined for an empty Storage"));
            }
            Ok(run_mean(mean_level, &values))
        })?,
    )?;

    let matmul_level = level;
    exports.set(
        "matmul_into",
        lua.create_function(move |_, args: Variadic<Value>| {
            let (lhs, rhs, out, m, k, n) = parse_matmul(args)?;
            matmul_into_values(matmul_level, lhs, rhs, out, m, k, n)
        })?,
    )?;

    let sgd_level = level;
    exports.set(
        "sgd_step",
        lua.create_function(move |_, args: Variadic<Value>| {
            let (parameter, gradient, learning_rate) = parse_sgd(args)?;
            sgd_step_values(sgd_level, parameter, gradient, learning_rate)
        })?,
    )?;

    Ok(())
}

fn argument_count(args: &Variadic<Value>, expected: usize, name: &str) -> Result<()> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(Error::runtime(format!(
            "{name} expects {expected} arguments, got {}",
            args.len()
        )))
    }
}

fn storage_argument(value: Value, name: &str) -> Result<AnyUserData> {
    match value {
        Value::UserData(userdata) if userdata.is::<Storage>() => Ok(userdata),
        Value::UserData(_) => Err(Error::runtime(format!("{name} must be a Storage"))),
        other => Err(Error::runtime(format!(
            "{name} must be a Storage, got {}",
            other.type_name()
        ))),
    }
}

fn parse_one_storage(args: Variadic<Value>, name: &str) -> Result<AnyUserData> {
    argument_count(&args, 1, name)?;
    storage_argument(
        args.into_iter().next().expect("checked argument count"),
        name,
    )
}

fn parse_binary(
    args: Variadic<Value>,
    name: &str,
) -> Result<(AnyUserData, AnyUserData, AnyUserData)> {
    argument_count(&args, 3, name)?;
    let mut args = args.into_iter();
    let lhs = storage_argument(
        args.next().expect("checked argument count"),
        &format!("{name} lhs"),
    )?;
    let rhs = storage_argument(
        args.next().expect("checked argument count"),
        &format!("{name} rhs"),
    )?;
    let out = storage_argument(
        args.next().expect("checked argument count"),
        &format!("{name} out"),
    )?;
    Ok((lhs, rhs, out))
}

/// Parse the documented `(input, scalar, out)` form and the equivalent
/// `(input, out, scalar)` compatibility form.
fn parse_scalar_unary(
    args: Variadic<Value>,
    name: &str,
) -> Result<(AnyUserData, f32, AnyUserData)> {
    argument_count(&args, 3, name)?;
    let mut args = args.into_iter();
    let input = storage_argument(
        args.next().expect("checked argument count"),
        &format!("{name} input"),
    )?;
    let second = args.next().expect("checked argument count");
    let third = args.next().expect("checked argument count");

    if matches!(&second, Value::UserData(userdata) if userdata.is::<Storage>()) {
        let out = storage_argument(second, &format!("{name} out"))?;
        let scalar = value_to_f32(third)?;
        Ok((input, scalar, out))
    } else {
        let scalar = value_to_f32(second)?;
        let out = storage_argument(third, &format!("{name} out"))?;
        Ok((input, scalar, out))
    }
}

fn parse_dimension(value: Value, name: &str) -> Result<usize> {
    let Value::Integer(value) = value else {
        return Err(Error::runtime(format!(
            "{name} must be a nonnegative Lua integer"
        )));
    };
    usize::try_from(value)
        .map_err(|_| Error::runtime(format!("{name} must be a nonnegative Lua integer")))
}

fn checked_product(lhs: usize, rhs: usize, name: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::runtime(format!("{name} dimensions overflow")))
}

fn parse_matmul(
    args: Variadic<Value>,
) -> Result<(AnyUserData, AnyUserData, AnyUserData, usize, usize, usize)> {
    argument_count(&args, 6, "matmul_into")?;
    let args: Vec<Value> = args.into_iter().collect();
    let lhs = storage_argument(args[0].clone(), "matmul_into lhs")?;
    let rhs = storage_argument(args[1].clone(), "matmul_into rhs")?;

    // Both `(lhs, rhs, m, k, n, out)` and `(lhs, rhs, out, m, k, n)`
    // are accepted; the first is the documented form because *_into returns
    // its final output argument.
    let (m, k, n, out) = if matches!(&args[2], Value::UserData(userdata) if userdata.is::<Storage>())
    {
        let out = storage_argument(args[2].clone(), "matmul_into out")?;
        let m = parse_dimension(args[3].clone(), "matmul_into m")?;
        let k = parse_dimension(args[4].clone(), "matmul_into k")?;
        let n = parse_dimension(args[5].clone(), "matmul_into n")?;
        (m, k, n, out)
    } else {
        let m = parse_dimension(args[2].clone(), "matmul_into m")?;
        let k = parse_dimension(args[3].clone(), "matmul_into k")?;
        let n = parse_dimension(args[4].clone(), "matmul_into n")?;
        let out = storage_argument(args[5].clone(), "matmul_into out")?;
        (m, k, n, out)
    };

    Ok((lhs, rhs, out, m, k, n))
}

fn parse_sgd(args: Variadic<Value>) -> Result<(AnyUserData, AnyUserData, f32)> {
    argument_count(&args, 3, "sgd_step")?;
    let mut args = args.into_iter();
    let parameter = storage_argument(
        args.next().expect("checked argument count"),
        "sgd_step parameter",
    )?;
    let gradient = storage_argument(
        args.next().expect("checked argument count"),
        "sgd_step gradient",
    )?;
    let learning_rate = value_to_f32(args.next().expect("checked argument count"))?;
    Ok((parameter, gradient, learning_rate))
}

fn storage_len(storage: &AnyUserData, name: &str) -> Result<usize> {
    let storage = storage
        .borrow::<Storage>()
        .map_err(|error| Error::runtime(format!("could not borrow {name}: {error}")))?;
    Ok(storage.as_slice().len())
}

fn ensure_len(storage: &AnyUserData, expected: usize, name: &str) -> Result<()> {
    let actual = storage_len(storage, name)?;
    if actual == expected {
        Ok(())
    } else {
        Err(Error::runtime(format!(
            "{name} has length {actual}, expected {expected}"
        )))
    }
}

fn snapshot_storage(storage: &AnyUserData, name: &str) -> Result<Vec<f32>> {
    let storage = storage
        .borrow::<Storage>()
        .map_err(|error| Error::runtime(format!("could not borrow {name}: {error}")))?;
    let mut snapshot = Vec::new();
    snapshot
        .try_reserve_exact(storage.as_slice().len())
        .map_err(|_| Error::runtime(format!("could not allocate {name} snapshot")))?;
    snapshot.extend_from_slice(storage.as_slice());
    Ok(snapshot)
}

fn binary_into(level: Level, args: Variadic<Value>, operation: BinaryOp) -> Result<AnyUserData> {
    let (lhs, rhs, out) = parse_binary(args, operation.name())?;
    let lhs_len = storage_len(&lhs, "binary lhs")?;
    ensure_len(&rhs, lhs_len, "binary rhs")?;
    ensure_len(&out, lhs_len, "binary out")?;

    // Snapshot all inputs before taking the mutable output borrow. This is
    // necessary when out aliases lhs, rhs, or both, and keeps every borrow
    // short and non-conflicting.
    let lhs_values = snapshot_storage(&lhs, "binary lhs")?;
    let rhs_values = snapshot_storage(&rhs, "binary rhs")?;
    let output = out.clone();
    {
        let mut out = out.borrow_mut::<Storage>().map_err(|error| {
            Error::runtime(format!("could not borrow binary out mutably: {error}"))
        })?;
        run_binary(
            level,
            operation,
            &lhs_values,
            &rhs_values,
            out.as_mut_slice(),
        );
    }
    // Inputs were snapshotted before this mutable borrow, so lhs/rhs may be
    // the same userdata as out (including both aliases).
    Ok(output)
}

fn unary_into(level: Level, args: Variadic<Value>, operation: UnaryOp) -> Result<AnyUserData> {
    argument_count(&args, 2, operation.name())?;
    let mut args = args.into_iter();
    let input = storage_argument(args.next().expect("checked argument count"), "unary input")?;
    let out = storage_argument(args.next().expect("checked argument count"), "unary out")?;
    unary_into_values(level, input, out, operation)
}

fn unary_into_values(
    level: Level,
    input: AnyUserData,
    out: AnyUserData,
    operation: UnaryOp,
) -> Result<AnyUserData> {
    let input_len = storage_len(&input, "unary input")?;
    ensure_len(&out, input_len, "unary out")?;
    let input_values = snapshot_storage(&input, "unary input")?;
    let output = out.clone();
    {
        let mut out = out.borrow_mut::<Storage>().map_err(|error| {
            Error::runtime(format!("could not borrow unary out mutably: {error}"))
        })?;
        run_unary(level, operation, &input_values, out.as_mut_slice());
    }
    Ok(output)
}

fn matmul_into_values(
    level: Level,
    lhs: AnyUserData,
    rhs: AnyUserData,
    out: AnyUserData,
    m: usize,
    k: usize,
    n: usize,
) -> Result<AnyUserData> {
    let lhs_len = checked_product(m, k, "lhs")?;
    let rhs_len = checked_product(k, n, "rhs")?;
    let out_len = checked_product(m, n, "output")?;
    ensure_len(&lhs, lhs_len, "matmul lhs")?;
    ensure_len(&rhs, rhs_len, "matmul rhs")?;
    ensure_len(&out, out_len, "matmul out")?;

    let lhs_values = snapshot_storage(&lhs, "matmul lhs")?;
    let rhs_values = snapshot_storage(&rhs, "matmul rhs")?;
    let output = out.clone();
    {
        let mut out = out.borrow_mut::<Storage>().map_err(|error| {
            Error::runtime(format!("could not borrow matmul out mutably: {error}"))
        })?;
        run_matmul(level, &lhs_values, &rhs_values, out.as_mut_slice(), m, k, n);
    }
    Ok(output)
}

fn sgd_step_values(
    level: Level,
    parameter: AnyUserData,
    gradient: AnyUserData,
    learning_rate: f32,
) -> Result<()> {
    let parameter_len = storage_len(&parameter, "sgd parameter")?;
    ensure_len(&gradient, parameter_len, "sgd gradient")?;

    // In particular, this snapshot makes `sgd_step(p, p, lr)` use every
    // original gradient value rather than values already updated in p.
    let gradient_values = snapshot_storage(&gradient, "sgd gradient")?;
    {
        let mut parameter = parameter.borrow_mut::<Storage>().map_err(|error| {
            Error::runtime(format!("could not borrow sgd parameter mutably: {error}"))
        })?;
        run_sgd(
            level,
            parameter.as_mut_slice(),
            &gradient_values,
            learning_rate,
        );
    }
    Ok(())
}

impl BinaryOp {
    fn name(self) -> &'static str {
        match self {
            Self::Add => "add_into",
            Self::Sub => "sub_into",
            Self::Mul => "mul_into",
        }
    }
}

impl UnaryOp {
    fn name(self) -> &'static str {
        match self {
            Self::Relu => "relu_into",
            Self::Scale(_) => "scale_into",
            Self::Pow(_) => "pow_into",
        }
    }
}

fn run_binary(level: Level, operation: BinaryOp, lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
    if level.is_fallback() {
        match operation {
            BinaryOp::Add => add_scalar(lhs, rhs, out),
            BinaryOp::Sub => sub_scalar(lhs, rhs, out),
            BinaryOp::Mul => mul_scalar(lhs, rhs, out),
        }
    } else {
        dispatch!(level, simd => binary_simd(simd, operation, lhs, rhs, out));
    }
}

fn run_unary(level: Level, operation: UnaryOp, input: &[f32], out: &mut [f32]) {
    if let UnaryOp::Pow(exponent) = operation {
        pow_scalar(input, exponent, out);
    } else if level.is_fallback() {
        match operation {
            UnaryOp::Relu => relu_scalar(input, out),
            UnaryOp::Scale(scale) => scale_scalar(input, scale, out),
            UnaryOp::Pow(_) => unreachable!("pow_into is scalar-only"),
        }
    } else {
        dispatch!(level, simd => unary_simd(simd, operation, input, out));
    }
}

fn run_sum(level: Level, input: &[f32]) -> f32 {
    if level.is_fallback() {
        sum_scalar(input)
    } else {
        dispatch!(level, simd => sum_simd(simd, input))
    }
}

fn run_mean(level: Level, input: &[f32]) -> f32 {
    if level.is_fallback() {
        mean_scalar(input)
    } else {
        dispatch!(level, simd => mean_simd(simd, input))
    }
}

fn run_matmul(
    level: Level,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    if level.is_fallback() {
        matmul_scalar(lhs, rhs, out, m, k, n);
    } else {
        dispatch!(level, simd => matmul_simd(simd, lhs, rhs, out, m, k, n));
    }
}

fn run_sgd(level: Level, parameter: &mut [f32], gradient: &[f32], learning_rate: f32) {
    if level.is_fallback() {
        sgd_scalar(parameter, gradient, learning_rate);
    } else {
        dispatch!(level, simd => sgd_simd(simd, parameter, gradient, learning_rate));
    }
}

#[inline(always)]
fn binary_simd<S: Simd>(simd: S, operation: BinaryOp, lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
    let lanes = S::f32s::N;
    let vector_end = lhs.len() / lanes * lanes;
    for start in (0..vector_end).step_by(lanes) {
        let lhs_vector = S::f32s::from_slice(simd, &lhs[start..start + lanes]);
        let rhs_vector = S::f32s::from_slice(simd, &rhs[start..start + lanes]);
        let result = match operation {
            BinaryOp::Add => lhs_vector + rhs_vector,
            BinaryOp::Sub => lhs_vector - rhs_vector,
            BinaryOp::Mul => lhs_vector * rhs_vector,
        };
        result.store_slice(&mut out[start..start + lanes]);
    }
    for index in vector_end..lhs.len() {
        out[index] = match operation {
            BinaryOp::Add => lhs[index] + rhs[index],
            BinaryOp::Sub => lhs[index] - rhs[index],
            BinaryOp::Mul => lhs[index] * rhs[index],
        };
    }
}

#[inline(always)]
fn unary_simd<S: Simd>(simd: S, operation: UnaryOp, input: &[f32], out: &mut [f32]) {
    let lanes = S::f32s::N;
    let vector_end = input.len() / lanes * lanes;
    for start in (0..vector_end).step_by(lanes) {
        let input_vector = S::f32s::from_slice(simd, &input[start..start + lanes]);
        let result = match operation {
            UnaryOp::Relu => {
                let zero = S::f32s::splat(simd, 0.0);
                input_vector.simd_gt(zero).select(input_vector, zero)
            }
            UnaryOp::Scale(scale) => input_vector * scale,
            // pow_into is intentionally scalar. This arm cannot be reached by
            // run_unary's SIMD path, but keeping the fallback here makes the
            // operation enum exhaustive and avoids silently changing pow.
            UnaryOp::Pow(_) => unreachable!("pow_into is scalar-only"),
        };
        result.store_slice(&mut out[start..start + lanes]);
    }
    for index in vector_end..input.len() {
        out[index] = match operation {
            UnaryOp::Relu => {
                if input[index] > 0.0 {
                    input[index]
                } else {
                    0.0
                }
            }
            UnaryOp::Scale(scale) => input[index] * scale,
            UnaryOp::Pow(_) => unreachable!("pow_into is scalar-only"),
        };
    }
}

#[inline(always)]
fn sum_simd<S: Simd>(simd: S, input: &[f32]) -> f32 {
    let lanes = S::f32s::N;
    let vector_end = input.len() / lanes * lanes;
    let mut accumulator = S::f32s::splat(simd, 0.0);
    for start in (0..vector_end).step_by(lanes) {
        accumulator += S::f32s::from_slice(simd, &input[start..start + lanes]);
    }
    let mut sum = 0.0;
    for value in accumulator.as_slice() {
        sum += *value;
    }
    for value in &input[vector_end..] {
        sum += *value;
    }
    sum
}

#[inline(always)]
fn mean_simd<S: Simd>(simd: S, input: &[f32]) -> f32 {
    sum_simd(simd, input) / input.len() as f32
}

#[inline(always)]
fn matmul_simd<S: Simd>(
    simd: S,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    out.fill(0.0);
    let lanes = S::f32s::N;
    let vector_end = n / lanes * lanes;
    for row in 0..m {
        let output_row = &mut out[row * n..(row + 1) * n];
        for inner in 0..k {
            let lhs_value = lhs[row * k + inner];
            let rhs_row = &rhs[inner * n..(inner + 1) * n];
            for start in (0..vector_end).step_by(lanes) {
                let current = S::f32s::from_slice(simd, &output_row[start..start + lanes]);
                let rhs_vector = S::f32s::from_slice(simd, &rhs_row[start..start + lanes]);
                (current + rhs_vector * lhs_value)
                    .store_slice(&mut output_row[start..start + lanes]);
            }
            for column in vector_end..n {
                output_row[column] += lhs_value * rhs_row[column];
            }
        }
    }
}

#[inline(always)]
fn sgd_simd<S: Simd>(simd: S, parameter: &mut [f32], gradient: &[f32], learning_rate: f32) {
    let lanes = S::f32s::N;
    let vector_end = parameter.len() / lanes * lanes;
    for start in (0..vector_end).step_by(lanes) {
        let parameter_vector = S::f32s::from_slice(simd, &parameter[start..start + lanes]);
        let gradient_vector = S::f32s::from_slice(simd, &gradient[start..start + lanes]);
        (parameter_vector - gradient_vector * learning_rate)
            .store_slice(&mut parameter[start..start + lanes]);
    }
    for index in vector_end..parameter.len() {
        parameter[index] -= learning_rate * gradient[index];
    }
}

pub(crate) fn add_scalar(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
    for ((out, lhs), rhs) in out.iter_mut().zip(lhs).zip(rhs) {
        *out = *lhs + *rhs;
    }
}

pub(crate) fn sub_scalar(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
    for ((out, lhs), rhs) in out.iter_mut().zip(lhs).zip(rhs) {
        *out = *lhs - *rhs;
    }
}

pub(crate) fn mul_scalar(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
    for ((out, lhs), rhs) in out.iter_mut().zip(lhs).zip(rhs) {
        *out = *lhs * *rhs;
    }
}

pub(crate) fn relu_scalar(input: &[f32], out: &mut [f32]) {
    for (out, input) in out.iter_mut().zip(input) {
        *out = if *input > 0.0 { *input } else { 0.0 };
    }
}

pub(crate) fn scale_scalar(input: &[f32], scale: f32, out: &mut [f32]) {
    for (out, input) in out.iter_mut().zip(input) {
        *out = *input * scale;
    }
}

pub(crate) fn pow_scalar(input: &[f32], exponent: f32, out: &mut [f32]) {
    for (out, input) in out.iter_mut().zip(input) {
        *out = input.powf(exponent);
    }
}

pub(crate) fn sum_scalar(input: &[f32]) -> f32 {
    input.iter().fold(0.0, |sum, value| sum + value)
}

pub(crate) fn mean_scalar(input: &[f32]) -> f32 {
    sum_scalar(input) / input.len() as f32
}

pub(crate) fn matmul_scalar(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    for row in 0..m {
        for column in 0..n {
            let mut value = 0.0;
            for inner in 0..k {
                value += lhs[row * k + inner] * rhs[inner * n + column];
            }
            out[row * n + column] = value;
        }
    }
}

pub(crate) fn sgd_scalar(parameter: &mut [f32], gradient: &[f32], learning_rate: f32) {
    for (parameter, gradient) in parameter.iter_mut().zip(gradient) {
        *parameter -= learning_rate * *gradient;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 2.0e-5_f32.max(expected.abs() * 2.0e-5);
            assert!(
                (actual - expected).abs() <= tolerance || (actual.is_nan() && expected.is_nan()),
                "index {index}: actual {actual:?}, expected {expected:?}, tolerance {tolerance}"
            );
        }
    }

    fn run_simd_binary(operation: BinaryOp, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        let level = Level::new();
        let mut out = vec![0.0; lhs.len()];
        if level.is_fallback() {
            binary_simd_for_test(level, operation, lhs, rhs, &mut out);
        } else {
            dispatch!(level, simd => binary_simd(simd, operation, lhs, rhs, &mut out));
        }
        out
    }

    fn binary_simd_for_test(
        level: Level,
        operation: BinaryOp,
        lhs: &[f32],
        rhs: &[f32],
        out: &mut [f32],
    ) {
        dispatch!(level, simd => binary_simd(simd, operation, lhs, rhs, out));
    }

    #[test]
    fn elementwise_simd_matches_scalar_for_lane_boundaries_and_tails() {
        for length in [
            0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 1003,
        ] {
            let lhs: Vec<f32> = (0..length)
                .map(|index| (index as f32 - 37.0) * 0.03125)
                .collect();
            let rhs: Vec<f32> = (0..length)
                .map(|index| (index as f32 + 11.0) * -0.0175)
                .collect();
            for operation in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
                let mut scalar = vec![0.0; length];
                match operation {
                    BinaryOp::Add => add_scalar(&lhs, &rhs, &mut scalar),
                    BinaryOp::Sub => sub_scalar(&lhs, &rhs, &mut scalar),
                    BinaryOp::Mul => mul_scalar(&lhs, &rhs, &mut scalar),
                }
                let simd = run_simd_binary(operation, &lhs, &rhs);
                assert_close(&simd, &scalar);
            }

            let mut scalar_relu = vec![0.0; length];
            relu_scalar(&lhs, &mut scalar_relu);
            let mut simd_relu = vec![0.0; length];
            let level = Level::new();
            if level.is_fallback() {
                relu_scalar(&lhs, &mut simd_relu);
            } else {
                dispatch!(level, simd => unary_simd(simd, UnaryOp::Relu, &lhs, &mut simd_relu));
            }
            assert_close(&simd_relu, &scalar_relu);

            let mut scalar_scale = vec![0.0; length];
            scale_scalar(&lhs, -1.75, &mut scalar_scale);
            let mut simd_scale = vec![0.0; length];
            let level = Level::new();
            if level.is_fallback() {
                scale_scalar(&lhs, -1.75, &mut simd_scale);
            } else {
                dispatch!(level, simd => unary_simd(simd, UnaryOp::Scale(-1.75), &lhs, &mut simd_scale));
            }
            assert_close(&simd_scale, &scalar_scale);
        }
    }

    #[test]
    fn reductions_match_with_large_odd_inputs() {
        let input: Vec<f32> = (0..100_003)
            .map(|index| ((index % 97) as f32 - 48.0) * 0.0078125)
            .collect();
        let scalar_sum = sum_scalar(&input);
        let level = Level::new();
        let simd_sum = if level.is_fallback() {
            scalar_sum
        } else {
            dispatch!(level, simd => sum_simd(simd, &input))
        };
        assert!((simd_sum - scalar_sum).abs() < 0.05);

        let scalar_mean = mean_scalar(&input);
        let simd_mean = if level.is_fallback() {
            scalar_mean
        } else {
            dispatch!(level, simd => mean_simd(simd, &input))
        };
        assert!((simd_mean - scalar_mean).abs() < 1.0e-5);
        assert_eq!(sum_scalar(&[]), 0.0);
    }

    #[test]
    fn matmul_simd_matches_scalar_for_rectangular_and_zero_shapes() {
        for &(m, k, n) in &[
            (1, 1, 1),
            (2, 3, 5),
            (3, 5, 2),
            (4, 7, 9),
            (5, 9, 17),
            (0, 4, 3),
            (4, 0, 3),
            (4, 3, 0),
        ] {
            let lhs: Vec<f32> = (0..m * k)
                .map(|index| (index as f32 - 4.0) * 0.125)
                .collect();
            let rhs: Vec<f32> = (0..k * n)
                .map(|index| (index as f32 + 2.0) * -0.0625)
                .collect();
            let mut scalar = vec![0.0; m * n];
            matmul_scalar(&lhs, &rhs, &mut scalar, m, k, n);
            let mut simd_output = vec![0.0; m * n];
            let level = Level::new();
            if level.is_fallback() {
                matmul_scalar(&lhs, &rhs, &mut simd_output, m, k, n);
            } else {
                dispatch!(level, simd => matmul_simd(
                    simd,
                    &lhs,
                    &rhs,
                    &mut simd_output,
                    m,
                    k,
                    n
                ));
            }
            assert_close(&simd_output, &scalar);
        }
    }

    #[test]
    fn sgd_simd_matches_scalar_and_uses_original_gradient() {
        let parameter: Vec<f32> = (0..1009).map(|index| index as f32 * 0.01).collect();
        let gradient: Vec<f32> = (0..1009)
            .map(|index| (index as f32 - 400.0) * 0.003)
            .collect();
        let mut scalar = parameter.clone();
        sgd_scalar(&mut scalar, &gradient, 0.125);
        let mut simd_parameter = parameter.clone();
        let level = Level::new();
        if level.is_fallback() {
            sgd_scalar(&mut simd_parameter, &gradient, 0.125);
        } else {
            dispatch!(level, simd => sgd_simd(
                simd,
                &mut simd_parameter,
                &gradient,
                0.125
            ));
        }
        assert_close(&simd_parameter, &scalar);

        let original = [2.0, -3.0, 4.0];
        let mut aliased = original;
        let gradient_snapshot = aliased;
        sgd_scalar(&mut aliased, &gradient_snapshot, 0.5);
        assert_eq!(aliased, [1.0, -1.5, 2.0]);
    }

    #[test]
    fn scalar_power_is_kept_separate_from_simd_kernels() {
        let input = [-2.0, -0.5, 0.0, 1.5, 3.0];
        let mut output = [0.0; 5];
        pow_scalar(&input, 3.0, &mut output);
        assert_close(&output, &[-8.0, -0.125, 0.0, 3.375, 27.0]);
    }
}
