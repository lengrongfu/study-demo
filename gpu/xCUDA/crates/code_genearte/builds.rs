extern crate bindgen;

use bindgen::callbacks::*;
use heck::ToSnakeCase;
use std::env;
use std::path::{Path, PathBuf};

macro_rules! p {
    ($($tokens: tt)*) => {
        println!("cargo:warning={}", format!($($tokens)*))
    }
}

// Install NVIDIA CUDA prior to building the bindings with `cargo build`.
// https://docs.rs/bindgen/latest/bindgen/struct.Builder.html
// TODO(sdake): I am hopeful we can autogenerate network objects as well via:
//              https://docs.rs/bindgen/latest/bindgen/callbacks/trait.ParseCallbacks.html
// TODO(sdake): We may want some variant of: .rustified_enum<T: AsRef<str>>(self, arg: T) -> Builder

#[derive(Debug)]
struct CB {}
impl ParseCallbacks for CB {
    fn item_name(&self, item: &str) -> Option<String> {
        // use heck::ToUpperCamelCase;
        Some(item.to_snake_case())
    }
}

fn generate_binding(header: String) -> bindgen::Bindings {
    let bindings = bindgen::Builder::default()
        .header(header)
        .derive_eq(true)
        .layout_tests(false) //不需要test,默认true
        .impl_debug(false) // 不实现debug
        .derive_debug(true)
        .array_pointers_in_arguments(true)
        // .parse_callbacks(Box::new(CB{}))
        .generate()
        .unwrap();
    bindings
}

fn cuda_build() {
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/");

    let bindings = generate_binding("/usr/local/cuda-11.8/include/cuda.h".to_string());

    let target_path = PathBuf::from("./src");
    bindings
        .write_to_file(target_path.join("cuda_driver_bindings_11_08.rs"))
        .expect("Couldn't write bindings!");

    p!(
        "Wrote bindings to {}",
        target_path.join("cuda_driver_bindings_11_08.rs").display()
    );
}

fn cuda_runtime_build() {
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/");

    let bindings = generate_binding("/usr/local/cuda-11.8/include/cuda_runtime_api.h".to_string());
    let target_path = PathBuf::from("./src");
    bindings
        .write_to_file(target_path.join("cuda_runtime_bindings_11_08.rs"))
        .expect("Couldn't write bindings!");

    p!(
        "Wrote bindings to {}",
        target_path.join("cuda_runtime_bindings_11_08.rs").display()
    );
}

fn nvml_build() {
    println!("cargo:rustc-link-lib=dylib=nvidia-ml");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/");

    let bindings = generate_binding("/usr/local/cuda-11.8/include/nvml.h".to_string());
    let target_path = PathBuf::from("./src");
    bindings
        .write_to_file(target_path.join("nvml_driver_bindings_11_08.rs"))
        .expect("Couldn't write bindings!");

    p!(
        "Wrote bindings to {}",
        target_path.join("nvml_driver_bindings_11_08.rs").display()
    );
}

fn main() {
    cuda_build();
    cuda_runtime_build();
    nvml_build();
}
