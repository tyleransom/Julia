% Simple dynamic example with asymmetric switching costs and expected wages

clear all; clc;
tic;
seed = 1234;
rand ('seed',seed);
randn('seed',seed);

N    = 1.95e5;
T    = 35;
time = ones(N,1)*[1:T];
nloc = 3;
J    = 2*nloc;

%% generate the data
% initial lagged choice and experience
LY     = unidrnd(J,N,1);
exper0 = zeros(N,1);

% amenities
alpha  = linspace(0,1,nloc)'; %location 1 is normalized to 0
alpha  = alpha(2:end);

% wages and unemployment benefits
bw = 0.4;
bb = 2.0;

% wage equation coefficients
bcons = 7.25;
bmed  = .10;
bbig  = .20;
bexp  = .06;
bexp2 = -0.0008;

% switching costs
bs = -2.0; %labor force status switching cost
bm = -6.0; %location moving cost

% discount factor
beta = .80; %take beta lower in simulations so finite dependence has some purchase. leave it higher when estimating on real data because higher is more realistic
% beta = 0;

% calculating the probability of each of the choices
a    = bs*kron(1-eye(J/nloc),ones(nloc));
am   = bm*kron(ones(J/nloc),1-eye(nloc));
fv   = zeros(N,J,max(exper0)+T+2,T+1);

for t=T:-1:1 % loop backwards through time
	for k=max(exper0)+T+1:-1:1 % loop over all possible values of experience
		for j=1:J % loop over previous decision
			fvtemp(:,1) = fv(:,1,k+1,t+1);
			fvtemp(:,2) = fv(:,2,k+1,t+1);
			fvtemp(:,3) = fv(:,3,k+1,t+1);
			fvtemp(:,4) = fv(:,4,k  ,t+1);
			fvtemp(:,5) = fv(:,5,k  ,t+1);
			fvtemp(:,6) = fv(:,6,k  ,t+1);
            dem = exp(0       +bw*(bcons        + bexp*(k-1) + bexp2*(k-1).^2)+a(1,j)+am(1,j)+fvtemp(:,1))+...
                  exp(alpha(1)+bw*(bcons + bmed + bexp*(k-1) + bexp2*(k-1).^2)+a(2,j)+am(2,j)+fvtemp(:,2))+...
                  exp(alpha(2)+bw*(bcons + bbig + bexp*(k-1) + bexp2*(k-1).^2)+a(3,j)+am(3,j)+fvtemp(:,3))+...
                  exp(0       +bb                                             +a(4,j)+am(4,j)+fvtemp(:,4))+...
                  exp(alpha(1)+bb                                             +a(5,j)+am(5,j)+fvtemp(:,5))+...
                  exp(alpha(2)+bb                                             +a(6,j)+am(6,j)+fvtemp(:,6));
            fv(:,j,k,t) = beta*(log(dem)-psi(1));
            p1(:,j,k,t) = exp(0       +bw*(bcons        + bexp*(k-1) + bexp2*(k-1).^2)+a(1,j)+am(1,j)+fvtemp(:,1))./dem;
            p2(:,j,k,t) = exp(alpha(1)+bw*(bcons + bmed + bexp*(k-1) + bexp2*(k-1).^2)+a(2,j)+am(2,j)+fvtemp(:,2))./dem;
            p3(:,j,k,t) = exp(alpha(2)+bw*(bcons + bbig + bexp*(k-1) + bexp2*(k-1).^2)+a(3,j)+am(3,j)+fvtemp(:,3))./dem;
            p4(:,j,k,t) = exp(0       +bb                                             +a(4,j)+am(4,j)+fvtemp(:,4))./dem;
            p5(:,j,k,t) = exp(alpha(1)+bb                                             +a(5,j)+am(5,j)+fvtemp(:,5))./dem;
            p6(:,j,k,t) = exp(alpha(2)+bb                                             +a(6,j)+am(6,j)+fvtemp(:,6))./dem;
			
            % using these probabilities to create the observed choices conditional on the lagged choices and wages
            draw=rand(N,1);
            Y(:,j,k,t)=(draw<p1(:,j,k,t)+p2(:,j,k,t)+p3(:,j,k,t)+p4(:,j,k,t)+p5(:,j,k,t)+p6(:,j,k,t))+...
					   (draw<            p2(:,j,k,t)+p3(:,j,k,t)+p4(:,j,k,t)+p5(:,j,k,t)+p6(:,j,k,t))+...
					   (draw<                        p3(:,j,k,t)+p4(:,j,k,t)+p5(:,j,k,t)+p6(:,j,k,t))+...
					   (draw<                                    p4(:,j,k,t)+p5(:,j,k,t)+p6(:,j,k,t))+...
					   (draw<                                                p5(:,j,k,t)+p6(:,j,k,t))+...
					   (draw<                                                            p6(:,j,k,t));
        end
	end
end

% using the lagged choices and the draws, creating the actual choices
Choice     = zeros(N,T);
actualFV   = zeros(N,T);
LY1        = LY;
exper      = zeros(N,T);
exper(:,1) = exper0+(LY>0 & LY<=nloc);

for t=1:T
    for j=1:J
        for k=1:max(exper0)+T+1
			Choice(:,t)   = Choice(:,t)+(LY1==j & exper(:,t)==k-1).*Y(:,j,k,t);
        end
    end
    LY1=Choice(:,t);
	if t<T
		exper(:,t+1)=exper(:,t)+(Choice(:,t)>0 & Choice(:,t)<=nloc);
	end
end

%create full lagged and future choices
LY=[LY Choice(:,1:T-1)];
FY=[Choice(:,2:T) Choice(:,T)];

%create wages
lnEarn(Choice>nloc) = NaN;
obsloc = NaN*ones(N,T);
for t=1:T
	obsloc(Choice(:,t)==1,t) = 1;
	obsloc(Choice(:,t)==2,t) = 2;
	obsloc(Choice(:,t)==3,t) = 3;
end

lnEarn = zeros(N,T);
for t=1:T
	lnEarn(:,t) = bcons + bmed*(Choice(:,t)==2) + bbig*(Choice(:,t)==3) + bexp*exper(:,t) + bexp2*exper(:,t).^2 + .3*randn(N,1);
end

%estimate wage equation
bhatw = regress(lnEarn(Choice<=nloc),[ones(sum(Choice(:)<=nloc),1) obsloc(Choice<=nloc)==2 obsloc(Choice<=nloc)==3 exper(Choice<=nloc) (exper(Choice<=nloc)).^2]);
[[bcons;bmed;bbig;bexp;bexp2] bhatw]

%estimate
LY1    = LY(:,1);
Y      = Choice;

tabulate(reshape(Choice(:,1:5),N*5,1));

b_ans=[alpha(1);0;alpha(2);0;0;bb;alpha(1);bb;alpha(2);bb;bw;bs;bm];

p1 = p1(:,:,1:8,1:7);
p2 = p2(:,:,1:8,1:7);
p3 = p3(:,:,1:8,1:7);
p4 = p4(:,:,1:8,1:7);
p5 = p5(:,:,1:8,1:7);
p6 = p6(:,:,1:8,1:7);
fv = fv(:,:,1:8,1:7);
if beta>0
    save data_SC_exper_wage_6_choices        b_ans Y LY FY lnEarn obsloc exper0 exper T N nloc time 
	save -v7.3 trueCCPs p1 p2 p3 p4 p5 p6
	save -v7.3 trueFVs fv
elseif beta==0
    save data_SC_exper_wage_6_choices_static b_ans Y LY FY lnEarn obsloc exper0 exper T N nloc time
	save -v7.3 trueCCPs_static p1 p2 p3 p4 p5 p6
	save -v7.3 trueFVs_static fv
end
% estimate dynamic model
% options=optimset('Disp','iter-detailed','FunValCheck','on','MaxFunEvals',1e8,'MaxIter',1e8);
% [b,like,e,o,g,h]=fminunc('est_SC_exper_wage_4_choices',b_ans,options,Y,LY,exper0,exper,collgrad,lngdp,age,T,N,nloc,bhatw);
% disp(['estimate ',' start val ',' truth ']);
% [b b_ans b_ans]
% [b sqrt(diag(inv(h)))]
% % test if estimates lie in 95% CI around true:
% in_true_CI = (abs((b-b_ans)./sqrt(diag(inv(h))))<1.96);
% [b b_ans in_true_CI]
disp(['Data generation took ',num2str(toc/60),' minutes'])
