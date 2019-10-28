mu = importdata('mu.txt');
sig = importdata('sig.txt');
w = importdata('w.txt');
mz = importdata('gmm_resampled_mz.txt');
% indexes of columns
data = 1:length(mu);

% filtracja na warianje
mdl.w = w;
mdl.mu = mu;
mdl.sig = sig;
clearvars h cv_x cv_y pdf_n cvxn BIC_all TiC_e TiC_est
CV=(mdl.sig).^2;

figure
h=histogram(CV);

cv_xn=h.BinEdges;
for i=1:size(cv_xn,2)-1
cvxn(i)=(cv_xn(i)+cv_xn(i+1))/2;
end
cv_x=cvxn';
cv_y=h.Values';
CVn=CV;
CVn(CVn>(quantile(CV,0.99)))=[];

clearvars h cv_x cv_y pdf_n cvxn BIC_all TiC_e TiC_est
figure
h=histogram(CVn);
title('Histogram of CV')
% xlabel('CV','FontSize',16,'FontWeight','bold')
% ylabel('Frequency','FontSize',16,'FontWeight','bold')
% set(gca,'FontSize',15)
cv_xn=h.BinEdges;
for i=1:size(cv_xn,2)-1
cvxn(i)=(cv_xn(i)+cv_xn(i+1))/2;
end
cv_x=cvxn';
cv_y=h.Values';
cv_x(cv_y==0)=[];
cv_y(cv_y==0)=[];


for i=3:10
[pp_e,mu_e,sig_e,TIC_e,l_lik_e]=gaussian_mixture_simple((CVn),ones(size(CVn)),i);
BIC_all(i)= -2*l_lik_e + (3*i-1)*log(sum(CVn));
end
BIC_all=BIC_all(3:10);
[~,BIC_f]=min(BIC_all);
[pp_est,mu_est,sig_est,TIC_est]=gaussian_mixture_simple((CVn),ones(size(CVn)),BIC_f+2);

for i=1:size(pp_est,1)
pdf_n(:,i)=(pp_est(i)*normpdf(cv_x,mu_est(i),sig_est(i)));
end
r1=pdf_n(:,size(pdf_n,2)); %7 6
r2=pdf_n(:,size(pdf_n,2)-1);
P = InterX([cv_x';r1'],[cv_x';r2']);
aaa=find((CV)>P(1,size(P,2)));

clearvars cv_x cv_y pdf_n cvxn BIC_all TiC_e TiC_est

w1=mdl.w; w1(aaa)=[];
mu1=mdl.mu; mu1(aaa)=[];
sig1=mdl.sig; sig1(aaa)=[];
data1=data; data1(:,aaa)=[];
% plot_gmm(mz,meanspec_base,w1,mu1,sig1);


% filtracja na amplitude
clearvars h cv_x cv_y pdf_n cvxn BIC_all TiC_e TiC_est w_est mu_est pp_est
for i=1:size(w1,1)
dc(i)=max(w1(i)*normpdf(mz,mu1(i),sig1(i)));
end
for i=1:size(w1,1)
dcnn(i)=max(w1(i)*normpdf(mz,mu1(i),sig1(i)));
end
dc=1./dc;
skal=linspace(1.51,1,size(dc,2));
figure
h=histogram(dc);
title('Histogram')
% xlabel('1/maximum value of component ','FontSize',16,'FontWeight','bold')
% ylabel('Frequency','FontSize',16,'FontWeight','bold')
% set(gca,'FontSize',15)
cv_xn=h.BinEdges;
for i=1:size(cv_xn,2)-1
cvxn(i)=(cv_xn(i)+cv_xn(i+1))/2;
end
cv_x=cvxn';
cv_y=h.Values';
cv_x(cv_y==0)=[];
cv_y(cv_y==0)=[];
[pp_est,mu_est,sig_est,TIC_est,l_lik_est]=gaussian_mixture_simple(dc,ones(size(dc)),1);
dcn=dc;
dcn(dcn>(quantile(dc,0.95)))=[];
clearvars h cv_x cv_y pdf_n cvxn BIC_all TiC_e TiC_est
figure
h=histogram(dcn);
title('max(pdf)')
cv_x=h.BinEdges(2:end)';
cv_xn=h.BinEdges;
for i=1:size(cv_xn,2)-1
cvxn(i)=(cv_xn(i)+cv_xn(i+1))/2;
end
cv_x=cvxn';
cv_y=h.Values';
cv_x(cv_y==0)=[];
cv_y(cv_y==0)=[];
for i=3:10
[pp_e,mu_e,sig_e,TIC_e,l_lik_e]=gaussian_mixture_simple((dcn),ones(size(dcn)),i);
BIC_all(i)= -2*l_lik_e + (3*i-1)*log(sum(dcn));
end
BIC_all=BIC_all(3:10);
[~,BIC_f]=min(BIC_all);
[pp_est,mu_est,sig_est,TIC_est]=gaussian_mixture_simple((dcn),ones(size(dcn)),BIC_f+2);
pdf_n=[];
for i=1:size(pp_est,1)
pdf_n(:,i)=(pp_est(i)*normpdf(cv_x,mu_est(i),sig_est(i)));
end
r1=pdf_n(:,4);
r2=pdf_n(:,3); % 1
P = InterX([cv_x';r1'],[cv_x';r2']);
a=find((dc)>P(1,1));
bbb=dcnn(a);
bb=max(bbb);

w2=w1; w2(a)=[];
mu2=mu1; mu2(a)=[];
sig2=sig1; sig2(a)=[];
data2=data1; data2(:,a)=[];

%

indices_after_variance = data1 - 1;
indices_after_both = data2 - 1;

dlmwrite('indices_after_variance.txt', indices_after_variance', 'newline', 'pc');
dlmwrite('indices_after_both.txt', indices_after_both', 'newline', 'pc');
