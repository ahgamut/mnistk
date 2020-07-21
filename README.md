# MNIST-1000 #

Package arranged to test 1001 different neural nets on the MNIST dataset.
Results are presented [here][webapp] (app loads slowly because of free dyno).

## Explanation

```
What if I had a bunch of different neural nets all trained on the same data?
```

The package contains 1001 different neural nets written in Pytorch, along with
scaffolding to test them on the MNIST dataset. The [`mnistk-webapp`][apprepo] 
repository contains a web application deployed to [Heroku][webapp], where you 
can see how each network performed.


[webapp]: https://mnistk.herokuapp.com
[apprepo]: https://github.com/ahgamut/mnistk-webapp
