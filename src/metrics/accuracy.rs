pub fn accuracy<T>(predictions: &[T], target: &[T]) -> f64
where
    T: Eq,
    T: PartialEq
{
    assert!(predictions.len() == target.len(), "Predictions and targets are of differing length, \
    cannot caclulcate accuracy");

    let no_correct: isize = predictions.iter().zip(target).map(|(p, t)| (p == t) as isize).sum();
    no_correct as f64 / predictions.len() as f64
}
