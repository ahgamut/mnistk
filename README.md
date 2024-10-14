# MNIST-1000 #

Package arranged to test 1001 different neural nets on the MNIST dataset.
I used to have a Heroku dyno running visualizations of the network, but that
expired `:(`, but for now you have rely on the summary blog posts:

https://ahgamut.github.io/2020/11/20/netpicking-1/
https://ahgamut.github.io/2020/11/24/netpicking-2/
https://ahgamut.github.io/2021/11/24/netpicking-3/

## Explanation

```
What if I had a bunch of different neural nets all trained on the same data?
```

The package contains 1001 different neural nets written in Pytorch, along with
scaffolding to test them on the MNIST dataset. The [`mnistk-webapp`][apprepo] 
repository contains a web application deployable to Heroku, to see how each
network performed.


[webapp]: https://mnistk.herokuapp.com
[apprepo]: https://github.com/ahgamut/mnistk-webapp
