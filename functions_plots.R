rm(list=ls())
workdir <- "/home/javi/Cloud/BGSE/term2/ML/roger/Machine-Learning/"
setwd(workdir)

loss_ev <- function(x){
  y=0
  for(i in 1:length(x))
  if(x[i]>=0){
    y[i]=1
  }else{
    y[i]=0
  }
  return(y)
}
relu <- function(x){
  y=0
  for(i in 1:length(x))
    if(x[i] < -1){
      y[i]=0
    }else{
      y[i]=x[i]+1
    }
  return(y)
}
loss_e <- function(x){
  return(exp(x))
}
loss_log <- function(x){
  return(log2(1 + exp(x)))
}

pdf("cost_functions.pdf")
plot(x=seq(-5,5, by=0.001),y=loss_ev(x=seq(-5,5, by=0.001)), type="l", ylim=c(0,3), 
     main="Cost functions", xlab="x",ylab="phi(x)")
lines(x=seq(-5,5, by=0.001), y=relu(x=seq(-5,5, by=0.001)), lty="dashed")
lines(x=seq(-5,5, by=0.001), y=loss_e(x=seq(-5,5, by=0.001)), lty="dotted")
lines(x=seq(-5,5, by=0.001), y=loss_log(x=seq(-5,5, by=0.001)), lty="twodash")
legend("topleft", lty=c(1,2,3,6), cex=0.9,
       legend=c("Expected Risk", "ReLU", "Exponential loss", "Logarithmic loss"))
dev.off()

###### margin loss
set.seed(333)
A <- cbind(rnorm(50, 0, 1), rbinom(50, size=1, prob=1))
B <- cbind(rnorm(50, 2, 1), rbinom(50, size=0, prob=1))

data <- as.data.frame(rbind(A, B))

eps("marginloss.eps")
ggplot(data=data, aes(y=data[,1], x=seq(1,100)))+
  geom_point(aes(shape=factor(data[,2])))+
  scale_shape_manual(values=c(1,3))+
  geom_abline(intercept=4.5, slope=-0.078)+
  geom_abline(intercept=3.8, slope=-0.078, linetype="dashed")+
  geom_abline(intercept=5.2, slope=-0.078, linetype="dashed")+
  guides(fill=FALSE)+
  ggtitle("Margin loss")+ylab(NULL)+xlab(NULL)+
  theme(legend.position="none")
dev.off()


