clc;
clear all;
close all;
n=300;
target=zeros(n,1);
output=zeros(n,1);
x1=unifrnd(-2*pi,2*pi, [n 1]);
y1=unifrnd(-2*pi,2*pi, [n 1]);
target=sin(x1)./x1.*sin(y1)./y1;

data=[x1 y1];
epoch=500;
number_train=round(0.7*n);
number_test=n-number_train;
number_mf=10;
Mean_mf_x=linspace(-2*pi,2*pi,number_mf);
Mean_mf_y=linspace(-2*pi,2*pi,number_mf);
Sigma_mf_x=unifrnd(1,2,[1,number_mf]);
Sigma_mf_y=unifrnd(1,2,[1,number_mf]);
rule=zeros(1,number_mf*number_mf);



eta=0.2;
eta_m=0.001;
number_parameter=3*number_mf;
f1=zeros(1,number_mf*number_mf);
parameter_c=unifrnd(-0.1,0.1,[number_mf*number_mf,3]);
for iter=1:epoch
   for i=1:number_train     
       x=data(i,1);
       y=data(i,2);
       x_input=unifrnd(x,x,[1,number_mf]);
       y_input=unifrnd(y,y,[1,number_mf]);
       sub_rule_x=exp(-0.5*((x_input-Mean_mf_x)./(Sigma_mf_x)).^2); 
       sub_rule_y=exp(-0.5*((y_input-Mean_mf_y)./(Sigma_mf_y)).^2);
       c=1;
       for p=1:number_mf
         for r=1:number_mf   
           rule(c)= sub_rule_x(p)*sub_rule_y(r);
           c=c+1;
         end
       end
       sum_rule=sum(rule);
       rule=rule/sum_rule;
       output=0;
       ss=1;
       for tr=1:number_mf*number_mf
           f1(ss)=parameter_c(ss,1)+parameter_c(ss,2)*x+parameter_c(ss,3)*y;
           output=output+f1(ss)*rule(ss);
           ss=ss+1;
       end
       
       output_final=output;
       %****************************
       error=target(i)-output_final;
       ss=1;
       for tr=1:number_mf*number_mf
          parameter_c(ss,1)=parameter_c(ss,1)+eta*error*rule(ss);
          parameter_c(ss,2)=parameter_c(ss,2)+eta*error*rule(ss)*x;
          parameter_c(ss,3)=parameter_c(ss,3)+eta*error*rule(ss)*y;
          ss=ss+1;
       end
%        for t=1:number_mf
%           Mean_mf(t)=Mean_mf(t)+eta_m*error*(x-Mean_mf(t))*rule(t)*f1(t);
%           Sigma_mf(t)=Sigma_mf(t)+(eta_m*error*(x-Mean_mf(t))^2*rule(t)*f1(t))/(Sigma_mf(t)^3);
%        end            
   end
   
 for i=1:number_train     
       x=data(i,1);
       y=data(i,2);
       x_input=unifrnd(x,x,[1,number_mf]);
       y_input=unifrnd(y,y,[1,number_mf]);
       sub_rule_x=exp(-0.5*((x_input-Mean_mf_x)./(Sigma_mf_x)).^2); 
       sub_rule_y=exp(-0.5*((y_input-Mean_mf_y)./(Sigma_mf_y)).^2);
       c=1;
       for p=1:number_mf
         for r=1:number_mf   
           rule(c)= sub_rule_x(p)*sub_rule_y(r);
           c=c+1;
         end
       end
       sum_rule=sum(rule);
       rule=rule/sum_rule;
       output1=0;
       ss=1;
       for tr=1:number_mf*number_mf
           f1(ss)=parameter_c(ss,1)+parameter_c(ss,2)*x+parameter_c(ss,3)*y;
           output1=output1+f1(ss)*rule(ss);
           ss=ss+1;
       end
       output(i)=output1
 end


figure(1);
subplot(1,2,1),plot(output(1:number_train),'.-b');
hold on;
subplot(1,2,1),plot(target(1:number_train),'.-r');
hold off;
mse1=mse(output(1:number_train)-target(1:number_train)');
title(sprintf('Sugeno Train - Sinc(x)*Sinc(y)\nEpoch = %d    MSE = %.10f ',iter,mse1),'fontsize',10,'fontweight','b');
legend('Sugeno Train','Target');

 for i=1:number_test     
       x=data(number_train+i,1);
       y=data(number_train+i,2);
       x_input=unifrnd(x,x,[1,number_mf]);
       y_input=unifrnd(y,y,[1,number_mf]);
       sub_rule_x=exp(-0.5*((x_input-Mean_mf_x)./(Sigma_mf_x)).^2); 
       sub_rule_y=exp(-0.5*((y_input-Mean_mf_y)./(Sigma_mf_y)).^2);
       c=1;
       for p=1:number_mf
         for r=1:number_mf   
           rule(c)= sub_rule_x(p)*sub_rule_y(r);
           c=c+1;
         end
       end
       sum_rule=sum(rule);
       rule=rule/sum_rule;
       output1=0;
       ss=1;
       for tr=1:number_mf*number_mf
           f1(ss)=parameter_c(ss,1)+parameter_c(ss,2)*x+parameter_c(ss,3)*y;
           output1=output1+f1(ss)*rule(ss);
           ss=ss+1;
       end
       output(number_train+i)=output1
 end


subplot(1,2,2),plot(output(number_train+1:n),'.-b');
hold on;
subplot(1,2,2),plot(target(number_train+1:n),'.-r');
hold off;
mse1=mse(output(number_train+1:n)-target(number_train+1:n)');
title(sprintf('Sugeno Test - Sinc(x)*Sinc(y)\n    MSE = %.10f ',mse1),'fontsize',10,'fontweight','b');
legend('Sugeno Test','Target');
end


