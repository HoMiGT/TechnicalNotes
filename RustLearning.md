> 1. 同一个作用域内，可以同时存在可变引用和不可引用，但是不可以交叉使用, <b style="color:red">二者的关系是互斥的</b>
>   * 编译可通过的版本
> ```rust
> let mut a = 10u32;
> let b = &mut a;
> *b = 2;
> let c = &a;
> println!("{c}");
> // *****************************
> let mut a = 10u32;
> let b = &mut a;
> *b = 2;
> let c = &a;
> println!("{a}");
> // ***************************
> let mut a = 10u32;
> let c = &a;
> let b = &mut a;
> *b = 2;
> println!("{b}");
> ```
