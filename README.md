# Adamant

A high-performance tensor computation and automatic differentiation library written in pure Rust.

[![Crates.io](https://img.shields.io/crates/v/adamant.svg)](https://crates.io/crates/adamant)
[![Documentation](https://docs.rs/adamant/badge.svg)](https://docs.rs/adamant)
[![Build Status](https://github.com/yourusername/adamant/workflows/Rust/badge.svg)](https://github.com/yourusername/adamant/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Adamant aims to provide a robust foundation for scientific computing, data analysis, and machine learning applications in Rust. It bridges the gap between the performance characteristics of established numerical libraries and the safety guarantees of the Rust programming language.

Features (in development):

- **N-dimensional tensor operations** with dynamic shapes
- **Automatic differentiation** for gradient-based optimization
- **SIMD optimizations** for maximum CPU performance
- **Clean, idiomatic Rust API** designed for safety and ergonomics
- **Comprehensive linear algebra** operations
- **No external dependencies** - pure Rust implementation

## Current Status

⚠️ **Early Development** ⚠️

Adamant is currently in the early planning and development stages. The API is subject to significant changes as we refine our approach.

## Installation

Once released, you'll be able to add Adamant to your Cargo.toml:

```toml
[dependencies]
adamant = "0.1.0"  # Placeholder version
```

## Example Usage

```rust
use adamant::Tensor;

// Create a 2x3 tensor filled with zeros
let a = Tensor::<f32>::new(&[2, 3]);

// Create a tensor from existing data
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let b = Tensor::from_vec(data, &[2, 3]).unwrap();

// Reshape a tensor
let c = b.reshape(&[3, 2]).unwrap();

// Access elements
let value = b.get(&[1, 2]).unwrap();
```

## Roadmap

- [x] Initial project structure
- [ ] Basic tensor implementation
- [ ] Core operations (add, multiply, etc.)
- [ ] Broadcasting support
- [ ] Linear algebra operations
- [ ] Automatic differentiation
- [ ] Performance optimizations
- [ ] Integration with deep learning frameworks

## Contributing

Contributions are welcome! As this is an early-stage project, please open an issue first to discuss any significant changes you'd like to make.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.
