function results = predictive_regression_ct(y, x, varargin)
%PREDICTIVE_REGRESSION_CT Predictive regression with Campbell-Thompson OOS stats.
%
%   RESULTS = PREDICTIVE_REGRESSION_CT(Y, X) estimates
%
%       y(t) = a + b * x(t-1) + u(t)
%
%   and returns:
%     - OLS coefficients
%     - conventional standard errors, t-stats, and p-values
%     - Newey-West standard errors, t-stats, and p-values
%     - in-sample R^2
%     - out-of-sample statistics versus the historical-average benchmark
%       using an expanding window in the spirit of Campbell and Thompson (2008)
%
%   X can either:
%     1) have the same length as Y, in which case the function uses X(1:end-1)
%        to predict Y(2:end), or
%     2) have length numel(Y)-1, in which case it is treated as already aligned
%        with Y(2:end).
%
%   Optional name-value pairs:
%     'NWLags'      : Newey-West lag length for in-sample inference.
%                     Default = floor(4 * (T / 100)^(2/9)).
%     'OOSStart'    : First out-of-sample observation index in the aligned sample.
%                     Must be at least 2. Default = max(20, floor(T/2)).
%     'WindowType'  : 'expanding' (default) or 'rolling'.
%     'WindowSize'  : Rolling window size. Required if WindowType = 'rolling'.
%     'OOSNWLags'   : Newey-West lag length for the Clark-West test.
%                     Default = 0 for one-step-ahead forecasts.
%     'ConstrainBeta': If true, impose the Campbell-Thompson sign restriction
%                      beta >= 0 in OOS forecasting. Default = false.
%
%   Example:
%       out = predictive_regression_ct(y, x, 'OOSStart', 60, 'NWLags', 4);
%       disp(out.summary)
%       disp(out.oos.summary)

    y = y(:);
    x = x(:);

    parser = inputParser;
    parser.FunctionName = mfilename;
    addParameter(parser, 'NWLags', []);
    addParameter(parser, 'OOSStart', []);
    addParameter(parser, 'WindowType', 'expanding');
    addParameter(parser, 'WindowSize', []);
    addParameter(parser, 'OOSNWLags', 0);
    addParameter(parser, 'ConstrainBeta', false);
    parse(parser, varargin{:});
    opts = parser.Results;

    if numel(x) == numel(y)
        Y = y(2:end);
        Xlag = x(1:end-1);
    elseif numel(x) == numel(y) - 1
        Y = y(2:end);
        Xlag = x;
    else
        error('X must have either the same length as Y or length(Y)-1.');
    end

    valid = isfinite(Y) & isfinite(Xlag);
    Y = Y(valid);
    Xlag = Xlag(valid);

    T = numel(Y);
    if T < 10
        error('Not enough usable observations after alignment. Need at least 10.');
    end

    if isempty(opts.NWLags)
        opts.NWLags = floor(4 * (T / 100)^(2 / 9));
    end
    opts.NWLags = max(0, floor(opts.NWLags));

    if isempty(opts.OOSStart)
        opts.OOSStart = max(2, min(T, max(20, floor(T / 2))));
    end
    opts.OOSStart = floor(opts.OOSStart);

    if opts.OOSStart < 2 || opts.OOSStart > T
        error('OOSStart must lie between 2 and the aligned sample size.');
    end

    X = [ones(T, 1), Xlag];
    [beta, se, tstat, pval, r2, residuals] = ols_inference(Y, X);
    [nwSe, nwTstat, nwPval] = newey_west_inference(Y, X, beta, opts.NWLags);

    coefNames = {'alpha'; 'beta'};
    summary = table(beta, se, tstat, pval, nwSe, nwTstat, nwPval, ...
        'VariableNames', {'coef', 'se', 'tstat', 'pvalue', ...
        'nw_se', 'nw_tstat', 'nw_pvalue'}, ...
        'RowNames', coefNames);

    oos = out_of_sample_stats(Y, Xlag, opts);

    results = struct();
    results.model = 'y(t) = alpha + beta * x(t-1) + u(t)';
    results.nobs = T;
    results.nw_lags = opts.NWLags;
    results.beta = beta(2);
    results.alpha = beta(1);
    results.se_beta = se(2);
    results.tstat_beta = tstat(2);
    results.pvalue_beta = pval(2);
    results.nw_se_beta = nwSe(2);
    results.nw_tstat_beta = nwTstat(2);
    results.nw_pvalue_beta = nwPval(2);
    results.R2 = r2;
    results.residuals = residuals;
    results.summary = summary;
    results.aligned_y = Y;
    results.aligned_x_lag = Xlag;
    results.oos = oos;
end

function [beta, se, tstat, pval, r2, residuals] = ols_inference(Y, X)
    [T, k] = size(X);
    beta = X \ Y;
    fitted = X * beta;
    residuals = Y - fitted;

    sse = residuals' * residuals;
    sst = sum((Y - mean(Y)).^2);
    r2 = 1 - sse / sst;

    sigma2 = sse / (T - k);
    XXinv = inv(X' * X);
    vcv = sigma2 * XXinv;
    se = sqrt(diag(vcv));
    tstat = beta ./ se;
    pval = 2 * (1 - student_t_cdf(abs(tstat), T - k));
end

function [nwSe, nwTstat, nwPval] = newey_west_inference(Y, X, beta, lags)
    [T, k] = size(X);
    residuals = Y - X * beta;
    XXinv = inv(X' * X);

    S = zeros(k, k);
    for t = 1:T
        xtu = X(t, :)' * residuals(t);
        S = S + xtu * xtu';
    end

    for ell = 1:lags
        weight = 1 - ell / (lags + 1);
        Gamma = zeros(k, k);
        for t = ell + 1:T
            xtu = X(t, :)' * residuals(t);
            xlu = X(t - ell, :)' * residuals(t - ell);
            Gamma = Gamma + xtu * xlu';
        end
        S = S + weight * (Gamma + Gamma');
    end

    vcvNW = XXinv * S * XXinv;
    nwSe = sqrt(diag(vcvNW));
    nwTstat = beta ./ nwSe;
    nwPval = 2 * (1 - normal_cdf(abs(nwTstat)));
end

function oos = out_of_sample_stats(Y, Xlag, opts)
    T = numel(Y);
    startIdx = opts.OOSStart;
    P = T - startIdx + 1;

    if P < 1
        error('OOSStart leaves no out-of-sample observations.');
    end

    yActual = zeros(P, 1);
    yHatModel = zeros(P, 1);
    yHatHist = zeros(P, 1);

    p = 0;
    for t = startIdx:T
        p = p + 1;

        if strcmpi(opts.WindowType, 'rolling')
            if isempty(opts.WindowSize)
                error('WindowSize must be supplied when WindowType is ''rolling''.');
            end
            windowSize = floor(opts.WindowSize);
            if windowSize < 5
                error('WindowSize must be at least 5.');
            end
            trainEnd = t - 1;
            trainStart = max(1, trainEnd - windowSize + 1);
        else
            trainStart = 1;
            trainEnd = t - 1;
        end

        yTrain = Y(trainStart:trainEnd);
        xTrain = Xlag(trainStart:trainEnd);

        Xtrain = [ones(numel(yTrain), 1), xTrain];
        betaTrain = Xtrain \ yTrain;

        if opts.ConstrainBeta
            betaTrain(2) = max(betaTrain(2), 0);
        end

        yActual(p) = Y(t);
        yHatHist(p) = mean(yTrain);
        yHatModel(p) = [1, Xlag(t)] * betaTrain;
    end

    errHist = yActual - yHatHist;
    errModel = yActual - yHatModel;

    msfeHist = mean(errHist .^ 2);
    msfeModel = mean(errModel .^ 2);
    r2OS = 1 - sum(errModel .^ 2) / sum(errHist .^ 2);

    cwSeries = errHist .^ 2 - (errModel .^ 2 - (yHatHist - yHatModel) .^ 2);
    [cwMean, cwSe] = nw_mean_stat(cwSeries, opts.OOSNWLags);
    cwTstat = cwMean / cwSe;
    cwPvalOneSided = 1 - normal_cdf(cwTstat);
    cwPvalTwoSided = 2 * (1 - normal_cdf(abs(cwTstat)));

    oosSummary = table(P, msfeModel, msfeHist, sqrt(msfeModel), sqrt(msfeHist), ...
        r2OS, cwTstat, cwPvalOneSided, cwPvalTwoSided, ...
        'VariableNames', {'n_oos', 'msfe_model', 'msfe_hist_avg', ...
        'rmse_model', 'rmse_hist_avg', 'R2_OS', 'cw_tstat', ...
        'cw_pvalue_onesided', 'cw_pvalue_twosided'});

    oos = struct();
    oos.window_type = lower(opts.WindowType);
    oos.oos_start = startIdx;
    oos.n_oos = P;
    oos.msfe_model = msfeModel;
    oos.msfe_hist_avg = msfeHist;
    oos.rmse_model = sqrt(msfeModel);
    oos.rmse_hist_avg = sqrt(msfeHist);
    oos.R2_OS = r2OS;
    oos.cw_tstat = cwTstat;
    oos.cw_pvalue_onesided = cwPvalOneSided;
    oos.cw_pvalue_twosided = cwPvalTwoSided;
    oos.actual = yActual;
    oos.forecast_model = yHatModel;
    oos.forecast_hist_avg = yHatHist;
    oos.error_model = errModel;
    oos.error_hist_avg = errHist;
    oos.summary = oosSummary;
end

function [mu, se] = nw_mean_stat(z, lags)
    z = z(:);
    z = z(isfinite(z));
    T = numel(z);

    mu = mean(z);
    centered = z - mu;

    gamma0 = sum(centered .* centered) / T;
    longRunVar = gamma0;

    for ell = 1:lags
        weight = 1 - ell / (lags + 1);
        gammaEll = sum(centered(ell + 1:end) .* centered(1:end - ell)) / T;
        longRunVar = longRunVar + 2 * weight * gammaEll;
    end

    se = sqrt(longRunVar / T);
end

function p = student_t_cdf(t, nu)
    t = real(t);
    p = zeros(size(t));

    for i = 1:numel(t)
        ti = t(i);
        xi = nu / (nu + ti ^ 2);
        ib = betainc(xi, nu / 2, 0.5);
        if ti >= 0
            p(i) = 1 - 0.5 * ib;
        else
            p(i) = 0.5 * ib;
        end
    end
end

function p = normal_cdf(z)
    p = 0.5 * erfc(-z ./ sqrt(2));
end
