function R = theta_registry()
% THETA_REGISTRY  Every mortality parameter vector used in the paper, in one place (R&R task 4, G5).
%
%   R = theta_registry();  theta = R.quad_HSLD.theta;
%
% theta = [theta0 (baseline log-odds), theta1 (time trend, on (d/Tmax)^2),
%          theta2 (seizure harm), theta3 (seizure x SOFA/10),
%          theta4 (drug harm),   theta5 (drug x Age/65)]
%
% Field .used_in lists the scripts and paper sections that use each vector,
% so the paper can state theta per experiment in one table. Values are
% copied from the scripts as they stood on 2026-10-06 (no values changed).

R.default.theta   = [-4, 0.5, 1.5, 1.0, 0.15, 0.07];
R.default.label   = 'Default';
R.default.short   = '\S5.3.1, \S5.4';
R.default.used_in = 's3, s4 (Sec 5.3.1 threshold sweep), s15 (Sec 5.4 EKF sensitivity)';

R.quad_HSLD.theta   = [-5, 0.5, 3.0, 1.0, 0.05, 0.02];
R.quad_HSLD.label   = 'High Sz / Low Drug';
R.quad_HSLD.short   = '\S5.2--5.3';
R.quad_HSLD.used_in = 's5, s8, s10-s14 (Secs 5.2-5.3)';

R.quad_HSHD.theta   = [-5, 0.5, 3.0, 1.0, 1.25, 0.3];
R.quad_HSHD.label   = 'High Sz / High Drug';
R.quad_HSHD.short   = '\S5.2--5.3';
R.quad_HSHD.used_in = 's5, s8, s10-s14 (Secs 5.2-5.3)';

R.quad_LSLD.theta   = [-5, 0.5, 0.5, 0.2, 0.05, 0.02];
R.quad_LSLD.label   = 'Low Sz / Low Drug';
R.quad_LSLD.short   = '\S5.2--5.3';
R.quad_LSLD.used_in = 's5, s8, s10-s14 (Secs 5.2-5.3)';

R.quad_LSHD.theta   = [-5, 0.5, 0.5, 0.2, 1.25, 0.3];
R.quad_LSHD.label   = 'Low Sz / High Drug';
R.quad_LSHD.short   = '\S5.2--5.3';
R.quad_LSHD.used_in = 's5, s8, s10-s14 (Secs 5.2-5.3)';

R.gformula.theta   = [-5, 0.5, 1.5, 0.5, 0.5, 0.2];
R.gformula.label   = 'G-formula';
R.gformula.short   = '\S5.1, \S5.5';
R.gformula.used_in = 's9 rise-decay DGP (Sec 5.1), s16 (Sec 5.5, incl. squared-burden generation)';

R.squared_gen.theta   = [-4, 0.5, 1.0, 0.7, 0.10, 0.05];
R.squared_gen.label   = 'Squared-burden generator';
R.squared_gen.short   = '\S5.5.3 (data generation)';
R.squared_gen.used_in = ['s16 Sec 5.5.3: GENERATES data with squared burden and these values, ', ...
                         'while the g-formula evaluates with R.gformula.theta (linear). ', ...
                         'The 14.5 pp bias therefore mixes wrong shape with wrong coefficient values.'];

R.gformula_sde.theta  = [-4, 0.8, 3.0, 0.8, 0.8, 0.3];
R.gformula_sde.label   = 'G-formula, SDE DGP';
R.gformula_sde.short   = 'not reported';
R.gformula_sde.used_in = 's9 with use_cicadas_dgp = true (March notes only; not in the submitted paper)';

end
