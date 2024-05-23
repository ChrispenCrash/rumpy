use ndarray::prelude::*;
use ndarray::{arr2, aview0, aview1, concatenate, stack, Array, Axis};
use std::f64::INFINITY as inf;

fn main() {
    //
    //
    //
    //
    // The basics
    println!("The Basics");
    println!("-----------");

    let a = array![[1., 2., 3.], [4., 5., 6.],];
    assert_eq!(a.ndim(), 2); // get the number of dimensions of array a
    assert_eq!(a.len(), 6); // get the number of elements in array a
    assert_eq!(a.shape(), [2, 3]); // get the shape of array a
    assert_eq!(a.is_empty(), false); // check if the array has zero elements

    println!("{:?}", a);

    //
    //
    //
    //
    // Array creation
    println!("\n\n\n\nArray creation");
    println!("--------------");

    println!("Element type and dimensionality");
    let a = Array::<f64, _>::zeros((3, 3).f());
    println!("{:?}", a);

    println!("Creating arrays with different initial values and/or different types");
    let a = Array::<bool, Ix2>::from_elem((3, 3), false);
    println!("{:?}", a);

    println!("Some common array initializing helper functions");
    let a = Array::<f64, _>::linspace(0., 5., 11);
    println!("{:?}", a);

    //
    //
    //
    //
    println!("\n\n\n\nBasic operations:");
    println!("----------------");
    //
    //
    let a = array![[10., 20., 30., 40.,],];
    let b = Array::range(0., 4., 1.); // [0., 1., 2., 3, ]

    assert_eq!(&a + &b, array![[10., 21., 32., 43.,]]); // Allocates a new array. Note the explicit `&`.
    assert_eq!(&a - &b, array![[10., 19., 28., 37.,]]);
    assert_eq!(&a * &b, array![[0., 20., 60., 120.,]]);
    assert_eq!(&a / &b, array![[inf, 20., 15., 13.333333333333334,]]);

    let a = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
    assert!(
        a.sum_axis(Axis(0)) == aview1(&[5., 7., 9.])
            && a.sum_axis(Axis(1)) == aview1(&[6., 15.])
            && a.sum_axis(Axis(0)).sum_axis(Axis(0)) == aview0(&21.)
            && a.sum_axis(Axis(0)).sum_axis(Axis(0)) == aview0(&a.sum())
    );

    println!("\nMatrix product");
    let a = array![[10., 20., 30., 40.,],];
    let b = Array::range(0., 4., 1.); // b = [0., 1., 2., 3, ]
    println!("a shape {:?}", &a.shape());
    println!("b shape {:?}", &b.shape());

    let b = b.into_shape((4, 1)).unwrap(); // reshape b to shape [4, 1]
    println!("b shape after reshape {:?}", &b.shape());

    println!("{}", a.dot(&b)); // [1, 4] x [4, 1] -> [1, 1]
    println!("{}", a.t().dot(&b.t())); // [4, 1] x [1, 4] -> [4, 4]

    //
    //
    //
    //
    println!("\n\n\n\nIndexing, Slicing and Iterating:");
    println!("--------------------------------");
    //
    //
    let a = Array::range(0., 10., 1.);

    let mut a = a.mapv(|a: f64| a.powi(3)); // numpy equivlant of `a ** 3`; https://doc.rust-lang.org/nightly/std/primitive.f64.html#method.powi

    println!("{}", a);

    println!("{}", a[[2]]);
    println!("{}", a.slice(s![2]));

    println!("{}", a.slice(s![2..5]));

    a.slice_mut(s![..6;2]).fill(1000.); // numpy equivlant of `a[:6:2] = 1000`
    println!("{}", a);

    for i in a.iter() {
        print!("{}, ", i.powf(1. / 3.))
    }

    let a = array![
        [
            [0, 1, 2], // a 3D array  2 x 2 x 3
            [10, 12, 13]
        ],
        [[100, 101, 102], [110, 112, 113]]
    ];

    let a = a.mapv(|a: isize| a.pow(1)); // numpy equivalent of `a ** 1`;
                                         // This line does nothing except illustrating mapv with isize type
    println!("a -> \n{}\n", a);

    println!("`a.slice(s![1, .., ..])` -> \n{}\n", a.slice(s![1, .., ..]));

    println!("`a.slice(s![.., .., 2])` -> \n{}\n", a.slice(s![.., .., 2]));

    println!(
        "`a.slice(s![.., 1, 0..2])` -> \n{}\n",
        a.slice(s![.., 1, 0..2])
    );

    println!("`a.iter()` ->");
    for i in a.iter() {
        print!("{}, ", i) // flat out to every element
    }

    println!("\n\n`a.outer_iter()` ->");
    for i in a.outer_iter() {
        print!("row: {}, \n", i) // iterate through first dimension
    }

    //
    //
    //
    //
    // Shape Manipulation
    println!("\n\n\n\nShape Manipulation:");
    println!("-------------------");
    //
    //

    println!("\nChanging the shape of an array");

    // Or you may use ndarray_rand crate to generate random arrays
    // let a = Array::random((2, 5), Uniform::new(0., 10.));

    let a = array![[3., 7., 3., 4.], [1., 4., 2., 2.], [7., 2., 4., 9.]];

    println!("a = \n{:?}\n", a);

    // use trait FromIterator to flatten a matrix to a vector
    let b = Array::from_iter(a.iter());
    println!("b = \n{:?}\n", b);

    let c = b.into_shape([6, 2]).unwrap(); // consume b and generate c with new shape
    println!("c = \n{:?}", c);

    // Stacking/concatenating together different arrays
    println!("\nStacking/concatenating together different arrays");
    let a = array![[3., 7., 8.], [5., 2., 4.],];

    let b = array![[1., 9., 0.], [5., 4., 1.],];

    println!("stack, axis 0:\n{:?}\n", stack![Axis(0), a, b]);
    println!("stack, axis 1:\n{:?}\n", stack![Axis(1), a, b]);
    println!("stack, axis 2:\n{:?}\n", stack![Axis(2), a, b]);
    println!("concatenate, axis 0:\n{:?}\n", concatenate![Axis(0), a, b]);
    println!("concatenate, axis 1:\n{:?}\n", concatenate![Axis(1), a, b]);

    // Splitting one array into several smaller ones
    println!("\nSplitting one array into several smaller ones");
    let a = array![
        [6., 7., 6., 9., 0., 5., 4., 0., 6., 8., 5., 2.],
        [8., 5., 5., 7., 1., 8., 6., 7., 1., 8., 1., 0.]
    ];

    let (s1, s2) = a.view().split_at(Axis(0), 1);
    println!("Split a from Axis(0), at index 1:");
    println!("s1  = \n{}", s1);
    println!("s2  = \n{}\n", s2);

    let (s1, s2) = a.view().split_at(Axis(1), 4);
    println!("Split a from Axis(1), at index 4:");
    println!("s1  = \n{}", s1);
    println!("s2  = \n{}\n", s2);

    //
    //
    //
    //
    // Copies and Views
    println!("\n\n\n\nCopies and Views:");
    println!("-----------------");
    //
    //

    println!("View, Ref or Shallow Copy");
    let mut a = Array::range(0., 12., 1.).into_shape([3, 4]).unwrap();
    println!("a = \n{}\n", a);

    {
        let (s1, s2) = a.view().split_at(Axis(1), 2);

        // with s as a view sharing the ref of a, we cannot update a here
        // a.slice_mut(s![1, 1]).fill(1234.);

        println!("Split a from Axis(0), at index 1:");
        println!("s1  = \n{}", s1);
        println!("s2  = \n{}\n", s2);
    }

    // now we can update a again here, as views of s1, s2 are dropped already
    a.slice_mut(s![1, 1]).fill(1234.);

    let (s1, s2) = a.view().split_at(Axis(1), 2);
    println!("Split a from Axis(0), at index 1:");
    println!("s1  = \n{}", s1);
    println!("s2  = \n{}\n", s2);

    println!("Deep Copy");
    let mut a = Array::range(0., 4., 1.).into_shape([2, 2]).unwrap();
    let b = a.clone();

    println!("a = \n{}\n", a);
    println!("b clone of a = \n{}\n", a);

    a.slice_mut(s![1, 1]).fill(1234.);

    println!("a updated...");
    println!("a = \n{}\n", a);
    println!("b clone of a = \n{}\n", b);

    //
    //
    //
    //
    println!("\n\n\n\nBroadcasting:");
    println!("-------------");
    //
    //
    let a = array![[1., 1.], [1., 2.], [0., 3.], [0., 4.]];

    let b = array![[0., 1.]];

    let c = array![[1., 2.], [1., 3.], [0., 4.], [0., 5.]];

    // We can add because the shapes are compatible even if not equal.
    // The `b` array is shape 1 × 2 but acts like a 4 × 2 array.
    assert!(c == a + b);

    let a = array![[1., 2.], [3., 4.],];

    let b = a.broadcast((3, 2, 2)).unwrap();
    println!("shape of a is {:?}", a.shape());
    println!("a is broadcased to 3x2x2 = \n{}", b);
}
