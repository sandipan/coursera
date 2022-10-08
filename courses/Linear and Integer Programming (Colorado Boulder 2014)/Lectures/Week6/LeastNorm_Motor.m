clear all 
clc
close all

R =1;

A = [1 0 -1  0;
     0 1  0 -1;
     R R  R  R];
 
F_d = [1
     1
     1];
 
f_hat = A\F_d
f_star = pinv(A)*F_d 

Z = [ 1;
     -1;
      1;
     -1]