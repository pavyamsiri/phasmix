extern crate alloc;

mod optimizer_config;

use indicatif::{ProgressBar, ProgressStyle};
use numpy::{PyArray1, PyReadonlyArray1};
use phasmix_core::{PSpiralComponent as RustComponent, PSpiralModel as RustModel};
use phasmix_fit::{GlobalOptimizer, PSpiralFitter as RustFitter, PSpiralFitterND};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use statrs::distribution::ContinuousCDF;

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct PSpiralComponent(pub RustComponent);

#[pymethods]
impl PSpiralComponent {
    #[new]
    fn new(
        alpha: f64,
        b: f64,
        c: f64,
        theta0: f64,
        scale_factor: f64,
        rho: f64,
        winding: i8,
    ) -> PyResult<Self> {
        let Ok(winding) = winding.try_into() else {
            return Err(PyValueError::new_err(format!(
                "winding must be -1 or 1, got {winding}"
            )));
        };
        Ok(Self(RustComponent {
            alpha,
            b_winding: b,
            c_winding: c,
            theta0,
            scale_factor,
            rho,
            winding,
            flattening_strength: 0.1,
        }))
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.0.alpha
    }
    #[getter]
    fn b(&self) -> f64 {
        self.0.b_winding
    }
    #[getter]
    fn c(&self) -> f64 {
        self.0.c_winding
    }
    #[getter]
    fn theta0(&self) -> f64 {
        self.0.theta0
    }
    #[getter]
    fn scale_factor(&self) -> f64 {
        self.0.scale_factor
    }
    #[getter]
    fn rho(&self) -> f64 {
        self.0.rho
    }
    #[getter]
    fn winding(&self) -> i8 {
        self.0.winding as i8
    }

    pub fn perturbation<'py>(
        &self,
        py: Python<'py>,
        z: PyReadonlyArray1<'py, f64>,
        vz: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let z = z.as_slice()?;
        let vz = vz.as_slice()?;
        let mut out = vec![0.0; z.len()];
        self.0.perturbation_vec(z, vz, &mut out);
        Ok(PyArray1::from_vec(py, out))
    }

    fn __repr__(&self) -> String {
        format!(
            "PSpiralComponent(alpha={:.4}, b={:.4}, c={:.4}, theta0={:.4}, scale_factor={:.4}, rho={:.4}, winding={})",
            self.0.alpha,
            self.0.b_winding,
            self.0.c_winding,
            self.0.theta0,
            self.0.scale_factor,
            self.0.rho,
            self.0.winding
        )
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct PSpiralModel(pub RustModel);

#[pymethods]
impl PSpiralModel {
    #[new]
    fn new(components: Vec<PSpiralComponent>) -> Self {
        Self(RustModel {
            components: components.into_iter().map(|c| c.0).collect(),
        })
    }

    #[getter]
    fn components(&self) -> Vec<PSpiralComponent> {
        self.0
            .components
            .iter()
            .map(|c| PSpiralComponent(c.clone()))
            .collect()
    }

    fn __repr__(&self) -> String {
        let comps: Vec<String> = self.components().iter().map(|c| c.__repr__()).collect();
        format!("PSpiralModel(components=[{}])", comps.join(", "))
    }

    pub fn perturbation<'py>(
        &self,
        py: Python<'py>,
        z: PyReadonlyArray1<'py, f64>,
        vz: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let z = z.as_slice()?;
        let vz = vz.as_slice()?;
        let mut out = vec![0.0; z.len()];
        self.0.perturbation_vec(z, vz, &mut out);
        Ok(PyArray1::from_vec(py, out))
    }
}

#[pyclass]
pub struct PSpiralFitResult {
    #[pyo3(get)]
    pub initial_model: PSpiralModel,
    #[pyo3(get)]
    pub final_model: PSpiralModel,
    #[pyo3(get)]
    pub data: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub initial_background: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub final_background: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub num_iterations: usize,
    #[pyo3(get)]
    pub max_iterations: Option<usize>,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub lnl: f64,
    #[pyo3(get)]
    pub initial_pvalue: f64,
    #[pyo3(get)]
    pub final_pvalue: f64,
    #[pyo3(get)]
    pub nfev: u64,
    #[pyo3(get)]
    pub nit: u64,
    #[pyo3(get)]
    pub optimizer_success: bool,
    #[pyo3(get)]
    pub optimizer_message: String,
    #[pyo3(get)]
    pub terminal: bool,
}

#[pyclass]
pub struct PSpiralFitIterator {
    inner: Option<phasmix_fit::PSpiralFitterIterative>,
    mask: Vec<f64>,
}

#[pymethods]
impl PSpiralFitIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PSpiralFitResult>> {
        let Some(iterator) = self.inner.as_mut() else {
            return Ok(None);
        };
        let Some(res) = py.detach(|| iterator.next()) else {
            self.inner = None;
            return Ok(None);
        };
        let dof = (6 * res.final_model.components.len()) as f64;
        let dist =
            statrs::distribution::ChiSquared::new(dof).expect("`freedom` is guaranteed positive.");
        let data = res.data.to_vec();
        let initial_null = phasmix_core::ln_likelihood(&data, &res.initial_background, &self.mask);
        let final_null = phasmix_core::ln_likelihood(&data, &res.final_background, &self.mask);
        Ok(Some(PSpiralFitResult {
            initial_model: PSpiralModel(res.initial_model),
            final_model: PSpiralModel(res.final_model),
            data: PyArray1::from_vec(py, data).into(),
            initial_background: PyArray1::from_vec(py, res.initial_background.to_vec()).into(),
            final_background: PyArray1::from_vec(py, res.final_background.to_vec()).into(),
            num_iterations: res.num_iterations,
            max_iterations: res.max_iterations,
            converged: res.converged,
            lnl: res.final_lnl,
            initial_pvalue: dist.sf(-2.0 * (initial_null - res.initial_lnl)),
            final_pvalue: dist.sf(-2.0 * (final_null - res.final_lnl)),
            nfev: res.nfev,
            nit: res.nit,
            optimizer_success: res.optimizer_success,
            optimizer_message: res.optimizer_message,
            terminal: res.terminal,
        }))
    }
}

#[pymethods]
impl PSpiralFitResult {
    fn __repr__(&self) -> String {
        format!(
            "PSpiralFitResult(num_iterations={}, converged={})",
            self.num_iterations, self.converged
        )
    }
}

#[pyclass]
pub struct PSpiralFitter {
    inner: RustFitter,
}

// The Python arrays are copied before detaching from the interpreter.
type BatchInput<'py> = (
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    (usize, usize),
);

impl PSpiralFitter {
    fn validate_options(
        num_components: Option<usize>,
        winding: Option<i8>,
    ) -> PyResult<Option<phasmix_core::Winding>> {
        if num_components.is_some_and(|count| count != 1 && count != 2) {
            return Err(PyValueError::new_err(
                "num_components must be 1, 2, or None",
            ));
        }
        winding
            .map(|value| {
                value
                    .try_into()
                    .map_err(|_| PyValueError::new_err("winding must be -1, 1, or None"))
            })
            .transpose()
    }

    fn convert_result(
        py: Python<'_>,
        res: phasmix_fit::PSpiralFitResult,
        mask: &[f64],
    ) -> PyResult<PSpiralFitResult> {
        let dof = (6 * res.final_model.components.len()) as f64;
        let dist =
            statrs::distribution::ChiSquared::new(dof).expect("`freedom` is guaranteed positive.");
        let lnl_initial_null =
            phasmix_core::ln_likelihood(&res.data, &res.initial_background, mask);
        let lnl_final_null = phasmix_core::ln_likelihood(&res.data, &res.final_background, mask);
        let lnl_initial = res.initial_lnl;
        let lnl_final = res.final_lnl;
        let lambda_initial = -2.0 * (lnl_initial_null - lnl_initial);
        let lambda_final = -2.0 * (lnl_final_null - lnl_final);

        let initial_pvalue = dist.sf(lambda_initial);
        let final_pvalue = dist.sf(lambda_final);

        Ok(PSpiralFitResult {
            initial_model: PSpiralModel(res.initial_model),
            final_model: PSpiralModel(res.final_model),
            data: PyArray1::from_vec(py, res.data.to_vec()).into(),
            initial_background: PyArray1::from_vec(py, res.initial_background.to_vec()).into(),
            final_background: PyArray1::from_vec(py, res.final_background.to_vec()).into(),
            num_iterations: res.num_iterations,
            max_iterations: res.max_iterations,
            converged: res.converged,
            lnl: res.final_lnl,
            initial_pvalue,
            final_pvalue,
            nfev: res.nfev,
            nit: res.nit,
            optimizer_success: res.optimizer_success,
            optimizer_message: res.optimizer_message,
            terminal: res.terminal,
        })
    }
}

impl PSpiralFitter {
    fn fitter_with_inputs(
        &self,
        count: Option<usize>,
        warm_start: Option<Vec<f64>>,
        seed: u64,
    ) -> PyResult<RustFitter> {
        if warm_start.is_none()
            && matches!(
                self.inner.fitter_single.optimizer,
                GlobalOptimizer::NelderMead { .. }
            )
        {
            return Err(PyValueError::new_err(
                "nelder_mead requires a warm_start and explicit component count",
            ));
        }
        let mut fitter = self.inner.clone();
        fitter.fitter_single.seed = Some(seed);
        fitter.fitter_double.seed = Some(seed);
        if warm_start.is_some() {
            let result = match count {
                Some(1) => fitter.fitter_single.set_warm_start(warm_start),
                Some(2) => fitter.fitter_double.set_warm_start(warm_start),
                _ => {
                    return Err(PyValueError::new_err(
                        "warm start requires an explicit component count",
                    ));
                }
            };
            result.map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        Ok(fitter)
    }
}

#[pymethods]
impl PSpiralFitter {
    #[new]
    #[pyo3(signature = (max_iterations=Some(50), atol=0.0, rtol=0.0, sigma_z=2.0, sigma_vz=2.0, *, bounds, optimizer))]
    fn new(
        max_iterations: Option<usize>,
        atol: f64,
        rtol: f64,
        sigma_z: f64,
        sigma_vz: f64,
        bounds: Vec<Vec<(f64, f64)>>,
        optimizer: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let optimizer_single = optimizer_config::build_optimizer::<6>(optimizer)?;
        let optimizer_double = optimizer_config::build_optimizer::<12>(optimizer)?;

        let mut component_bounds = bounds;
        if component_bounds.is_empty() || component_bounds.len() > 2 {
            return Err(PyValueError::new_err(
                "bounds must contain one or two component bound sets",
            ));
        }
        if component_bounds.len() == 1 {
            component_bounds.push(component_bounds[0].clone());
        }
        if component_bounds.iter().any(|component| {
            component.len() != 6
                || component
                    .iter()
                    .any(|(lower, upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
        }) {
            return Err(PyValueError::new_err(
                "each component must have six finite, ordered parameter bounds",
            ));
        }
        let bounds_for = |component: &[(f64, f64)]| {
            (
                component[0],
                component[1],
                component[2],
                component[3],
                component[4],
                component[5],
            )
        };
        let single = bounds_for(&component_bounds[0]);
        let double = bounds_for(&component_bounds[1]);

        let mut fitter = Self {
            inner: RustFitter {
                fitter_single: PSpiralFitterND {
                    warm_start: None,
                    seed: None,
                    parameter_bounds: None,
                    optimizer: optimizer_single,
                    alpha_bounds: single.0,
                    b_bounds: single.1,
                    c_bounds: single.2,
                    theta0_bounds: single.3,
                    scale_factor_bounds: single.4,
                    rho_bounds: single.5,
                },
                fitter_double: PSpiralFitterND {
                    warm_start: None,
                    seed: None,
                    parameter_bounds: None,
                    optimizer: optimizer_double,
                    alpha_bounds: double.0,
                    b_bounds: double.1,
                    c_bounds: double.2,
                    theta0_bounds: double.3,
                    scale_factor_bounds: double.4,
                    rho_bounds: double.5,
                },
                max_iterations,
                sigma_z,
                sigma_vz,
                atol,
                rtol,
            },
        };
        fitter.update_bounds(component_bounds)?;
        Ok(fitter)
    }

    /// Validate and atomically replace component bounds on the native fitter.
    pub fn update_bounds(&mut self, bounds: Vec<Vec<(f64, f64)>>) -> PyResult<()> {
        if bounds.is_empty()
            || bounds.len() > 2
            || bounds.iter().any(|component| component.len() != 6)
        {
            return Err(PyValueError::new_err(
                "bounds must contain one or two sets of six parameter bounds",
            ));
        }
        let single = bounds[0].clone();
        let mut double = single.clone();
        double.extend_from_slice(bounds.get(1).unwrap_or(&bounds[0]));
        let mut inner = self.inner.clone();
        inner
            .fitter_single
            .update_bounds(single)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        inner
            .fitter_double
            .update_bounds(double)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.inner = inner;
        Ok(())
    }

    /// Copy the native fitter with one local Nelder–Mead search per winding candidate.
    #[pyo3(signature = (*, maxiter=1500))]
    pub fn with_local_optimizer(&self, maxiter: usize) -> PyResult<Self> {
        if maxiter == 0 {
            return Err(PyValueError::new_err("maxiter must be positive"));
        }
        let mut inner = self.inner.clone();
        inner.fitter_single.optimizer = GlobalOptimizer::NelderMead { max_iter: maxiter };
        inner.fitter_double.optimizer = GlobalOptimizer::NelderMead { max_iter: maxiter };
        Ok(Self { inner })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "API would be overly complicated in order to reduce number of arguments."
    )]
    #[pyo3(signature = (initial_density, initial_background, mask, mesh_x, mesh_y, shape, *, seed, num_components=None, winding=None, improve_background=true, warm_start=None))]
    pub fn fit_spiral_with_background<'py>(
        &self,
        py: Python<'py>,
        initial_density: PyReadonlyArray1<'py, f64>,
        initial_background: PyReadonlyArray1<'py, f64>,
        mask: PyReadonlyArray1<'py, f64>,
        mesh_x: PyReadonlyArray1<'py, f64>,
        mesh_y: PyReadonlyArray1<'py, f64>,
        shape: (usize, usize),
        seed: u64,
        num_components: Option<usize>,
        winding: Option<i8>,
        improve_background: bool,
        warm_start: Option<Vec<f64>>,
    ) -> PyResult<PSpiralFitResult> {
        let winding = Self::validate_options(num_components, winding)?;
        let fitter = self.fitter_with_inputs(num_components, warm_start, seed)?;
        let initial_density = initial_density.as_slice()?.to_vec();
        let initial_background = initial_background.as_slice()?.to_vec();
        let mask = mask.as_slice()?.to_vec();
        let mesh_x = mesh_x.as_slice()?.to_vec();
        let mesh_y = mesh_y.as_slice()?.to_vec();

        // Own NumPy inputs before releasing the GIL for the full native fit.
        let res = py.detach(|| {
            fitter
                .fit_spiral_with_background_iterative(
                    &initial_density,
                    &initial_background,
                    &mask,
                    &mesh_x,
                    &mesh_y,
                    shape,
                    num_components,
                    winding,
                    improve_background,
                )
                .last()
                .ok_or_else(|| PyValueError::new_err("fit produced no checkpoints"))
        })?;

        Self::convert_result(py, res, &mask)
    }

    /// Copy a batch into Rust storage, then fit inside one shared Rayon pool.
    #[pyo3(signature = (inputs, *, seeds, workers=None, options=None, warm_starts=None, progress=false))]
    pub fn fit_batch(
        &self,
        py: Python<'_>,
        inputs: Vec<BatchInput<'_>>,
        seeds: Vec<u64>,
        workers: Option<usize>,
        options: Option<Vec<(Option<usize>, Option<i8>, bool)>>,
        warm_starts: Option<Vec<Option<Vec<f64>>>>,
        progress: bool,
    ) -> PyResult<Vec<PSpiralFitResult>> {
        if workers == Some(0) {
            return Err(PyValueError::new_err("workers must be positive"));
        }
        if seeds.len() != inputs.len() {
            return Err(PyValueError::new_err(
                "seeds must match the number of inputs",
            ));
        }
        let options = options.unwrap_or_else(|| vec![(None, None, true); inputs.len()]);
        if options.len() != inputs.len() {
            return Err(PyValueError::new_err(
                "options must match the number of inputs",
            ));
        }
        let options = options
            .into_iter()
            .map(|(count, winding, improve)| {
                Ok((count, Self::validate_options(count, winding)?, improve))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let warm_starts = warm_starts.unwrap_or_else(|| vec![None; inputs.len()]);
        if warm_starts.len() != inputs.len() {
            return Err(PyValueError::new_err(
                "warm_starts must match the number of inputs",
            ));
        }
        let fitters = options
            .iter()
            .zip(warm_starts)
            .zip(seeds)
            .map(|((&(count, _, _), start), seed)| self.fitter_with_inputs(count, start, seed))
            .collect::<PyResult<Vec<_>>>()?;
        let owned = inputs
            .into_iter()
            .map(|(data, background, mask, x, y, shape)| {
                let arrays = [data, background, mask, x, y];
                let size = shape
                    .0
                    .checked_mul(shape.1)
                    .filter(|size| *size > 0)
                    .ok_or_else(|| {
                        PyValueError::new_err("batch shapes must be nonempty and not overflow")
                    })?;
                let copies = arrays
                    .iter()
                    .map(|array| {
                        let slice = array.as_slice()?;
                        if slice.len() != size || slice.iter().any(|value| !value.is_finite()) {
                            return Err(PyValueError::new_err(
                                "batch arrays must be finite and match their shape",
                            ));
                        }
                        Ok(slice.to_vec())
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok((copies, shape))
            })
            .collect::<PyResult<Vec<_>>>()?;
        if owned.is_empty() {
            return Ok(Vec::new());
        }
        let results = py.detach(|| {
            let mut builder = rayon::ThreadPoolBuilder::new();
            if let Some(workers) = workers {
                builder = builder.num_threads(workers);
            }
            let pool = builder
                .build()
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let bar = if progress {
                ProgressBar::new(owned.len() as u64).with_style(
                    ProgressStyle::with_template(
                        "Fitting batch [{bar:40.cyan/blue}] {pos}/{len} [{elapsed_precise}]",
                    )
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
                )
            } else {
                ProgressBar::hidden()
            };
            bar.tick();
            let results = pool.install(|| {
                owned
                    .par_iter()
                    .zip(&options)
                    .zip(&fitters)
                    .map(|(((arrays, shape), &(count, winding, improve)), fitter)| {
                        let result = fitter
                            .fit_spiral_with_background_iterative(
                                &arrays[0], &arrays[1], &arrays[2], &arrays[3], &arrays[4], *shape,
                                count, winding, improve,
                            )
                            .last()
                            .ok_or_else(|| PyValueError::new_err("fit produced no checkpoints"));
                        bar.inc(1);
                        result
                    })
                    .collect::<PyResult<Vec<_>>>()
            });
            if results.is_ok() {
                bar.finish();
            } else {
                bar.abandon();
            }
            Ok::<_, PyErr>(results)
        })??;
        results
            .into_iter()
            .zip(&owned)
            .map(|(result, (arrays, _))| Self::convert_result(py, result, &arrays[2]))
            .collect()
    }

    /// Run the Rust refinement iterator and return every accepted checkpoint.
    #[expect(
        clippy::too_many_arguments,
        reason = "This mirrors the batch fitting API for event checkpoints."
    )]
    #[pyo3(signature = (initial_density, initial_background, mask, mesh_x, mesh_y, shape, *, seed, num_components=None, winding=None, improve_background=true, warm_start=None))]
    pub fn fit_spiral_with_background_events(
        &self,
        initial_density: PyReadonlyArray1<'_, f64>,
        initial_background: PyReadonlyArray1<'_, f64>,
        mask: PyReadonlyArray1<'_, f64>,
        mesh_x: PyReadonlyArray1<'_, f64>,
        mesh_y: PyReadonlyArray1<'_, f64>,
        shape: (usize, usize),
        seed: u64,
        num_components: Option<usize>,
        winding: Option<i8>,
        improve_background: bool,
        warm_start: Option<Vec<f64>>,
    ) -> PyResult<PSpiralFitIterator> {
        let winding = Self::validate_options(num_components, winding)?;
        let fitter = self.fitter_with_inputs(num_components, warm_start, seed)?;
        let initial_density = initial_density.as_slice()?;
        let initial_background = initial_background.as_slice()?;
        let mask = mask.as_slice()?;
        let mesh_x = mesh_x.as_slice()?;
        let mesh_y = mesh_y.as_slice()?;
        let results = fitter.fit_spiral_with_background_iterative(
            initial_density,
            initial_background,
            mask,
            mesh_x,
            mesh_y,
            shape,
            num_components,
            winding,
            improve_background,
        );
        Ok(PSpiralFitIterator {
            inner: Some(results),
            mask: mask.to_vec(),
        })
    }
}

#[pyfunction]
fn ln_likelihood(
    data: PyReadonlyArray1<f64>,
    prediction: PyReadonlyArray1<f64>,
    mask: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let data = data.as_slice()?;
    let prediction = prediction.as_slice()?;
    let mask = mask.as_slice()?;

    Ok(phasmix_core::ln_likelihood(data, prediction, mask))
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ln_likelihood, m)?)?;
    m.add_class::<PSpiralComponent>()?;
    m.add_class::<PSpiralModel>()?;
    m.add_class::<PSpiralFitter>()?;
    m.add_class::<PSpiralFitResult>()?;
    m.add_class::<PSpiralFitIterator>()?;
    Ok(())
}
