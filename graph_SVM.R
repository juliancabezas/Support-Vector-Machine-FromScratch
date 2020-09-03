library(e1071)

set.seed(10)
x1 = matrix(rnorm(40,1,1), 10, 2)
x2 = matrix(rnorm(40,4,1), 10, 2)
x <-rbind(x1,x2)
x
y = rep(c(-1, 1), c(10, 10))
y
#x[y == 1,] = x[y == 1,] + 1
plot(x, col = y + 3, pch = 19)


dat = data.frame(x, y = as.factor(y))
dat
svmfit = svm(y ~ ., data = dat, kernel = "linear", cost = 10, scale = FALSE)
print(svmfit)

make.grid = function(x, n = 75) {
  grange = apply(x, 2, range)
  x1 = seq(from = grange[1,1], to = grange[2,1], length = n)
  x2 = seq(from = grange[1,2], to = grange[2,2], length = n)
  expand.grid(X1 = x1, X2 = x2)
}

xgrid = make.grid(x)
xgrid[1:10,]


ygrid = predict(svmfit, xgrid)
plot(xgrid, col = c("red","blue")[as.numeric(ygrid)], pch = 20, cex = .2)
points(x, col = y + 3, pch = 19)
points(x[svmfit$index,], pch = 5, cex = 2)

beta = drop(t(svmfit$coefs)%*%x[svmfit$index,])
beta0 = svmfit$rho

pdf(file = "Document_latex/margin.pdf")

plot(xgrid, col = c("red", "dodgerblue4")[as.numeric(ygrid)], pch = 20, cex = .1)
points(x, col = y + 3, pch = 19)
points(x[svmfit$index,], pch = 5, cex = 2)
abline(beta0 / beta[2], -beta[1] / beta[2])
abline((beta0 - 1) / beta[2], -beta[1] / beta[2], lty = 2)
abline((beta0 + 1) / beta[2], -beta[1] / beta[2], lty = 2)
segments(2,0.85,3.78,2.71,lwd=2,col="black")
text(2.6,1.9,"Margin",cex=1,srt=45)

dev.off()



library(e1071)

set.seed(1015)
x = matrix(rnorm(40), 20, 2)

x1 = matrix(rnorm(40,2,1), 10, 2)
x2 = matrix(rnorm(40,3,1), 10, 2)
x <-rbind(x1,x2)

y = rep(c(-1, 1), c(10, 10))
x[y == 1,] = x[y == 1,] + 1
plot(x, col = y + 3, pch = 19)


dat = data.frame(x, y = as.factor(y))
dat
svmfit = svm(y ~ ., data = dat, kernel = "linear", cost = 10, scale = FALSE)
print(svmfit)

make.grid = function(x, n = 75) {
  grange = apply(x, 2, range)
  x1 = seq(from = grange[1,1], to = grange[2,1], length = n)
  x2 = seq(from = grange[1,2], to = grange[2,2], length = n)
  expand.grid(X1 = x1, X2 = x2)
}

xgrid = make.grid(x)
xgrid[1:10,]


ygrid = predict(svmfit, xgrid)
plot(xgrid, col = c("red","blue")[as.numeric(ygrid)], pch = 20, cex = .2)
points(x, col = y + 3, pch = 19)
points(x[svmfit$index,], pch = 5, cex = 2)

beta = drop(t(svmfit$coefs)%*%x[svmfit$index,])
beta0 = svmfit$rho

pdf(file = "Document_latex/margin_sf.pdf")

plot(xgrid, col = c("red", "blue")[as.numeric(ygrid)], pch = 20, cex = .2)
points(x, col = y + 3, pch = 19)
points(x[svmfit$index,], pch = 5, cex = 2)
abline(beta0 / beta[2], -beta[1] / beta[2])
abline((beta0 - 1) / beta[2], -beta[1] / beta[2], lty = 2)
abline((beta0 + 1) / beta[2], -beta[1] / beta[2], lty = 2)
segments(3.8,1.75,4.32,2.68,lwd=2,col="black")
text(3.9,2.25,"Margin",cex=1,srt=60)

dev.off()


