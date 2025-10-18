% Dragoș - Alexandru Duduc, grupa 324AA, 27 aprilie 2025
%% CITIREA DATELOR SI INITIALIZARI

clear;
clc;

% Extrag datele din fisierul .csv
opts = detectImportOptions('AirQualityUCI.csv', 'Delimiter',';', 'DecimalSeparator',',', 'VariableNamesLine',1, 'VariableNamingRule','preserve');
opts.DataLines = [2, 350];
opts.SelectedVariableNames = opts.VariableNames(3:end);
setDate = readtable('AirQualityUCI.csv', opts);
setDate = table2array(setDate);

% Elimin seturile care au lipsuri in vectorul de etichete
valid = setDate(:,1) ~= -200;
setDate = setDate(valid, :);

% Izolez vectorul de etichete
e = setDate(:,1);
setDate = setDate(:, 2:end);

[nrTotalExemple, nrCaract] = size(setDate);

% Impart baza de date in baza de antrenare si de testare
procentAntrenare = 0.8;
procentTest = 1 - procentAntrenare;
nrExempleAntrenament = floor(nrTotalExemple * procentAntrenare);
nrExempleTest = nrTotalExemple - nrExempleAntrenament;

A = setDate(1 : nrExempleAntrenament, :);
exempleTest = setDate(nrExempleAntrenament + 1 : end, :);
eAntrenament = e(1:nrExempleAntrenament);
eTest = e(nrExempleAntrenament + 1 : end);

% Inlocuiesc lipsurile cu valoarea medie a caracteristicii
A(A == -200) = NaN;
medie = mean(A, 1, 'omitnan');
for j = 1:nrCaract
    poz = isnan(A(:,j));
    A(poz,j) = medie(j);
end

exempleTest(exempleTest == -200) = NaN;
for j = 1:nrCaract
    poz = isnan(exempleTest(:,j));
    exempleTest(poz,j) = medie(j);
end

% Standardizare
mu    = mean(A, 1);
sigma = std( A, [], 1);

A = (A - mu) ./ sigma;
exempleTest = (exempleTest - mu) ./ sigma;

% Adaug coloana de 1 pentru bias
AExtins = [A, ones(nrExempleAntrenament,1)];
exempleTest = [exempleTest, ones(nrExempleTest,1)];

% Initializari
a = 1;
g = @(z) a * (z.^3);
g_prim = @(z) 3 * a * (z.^2);
g_secund = @(z) 6 * a * z;

nrNeuroni = 20;

rng(0);
x = randn(nrNeuroni, 1) * 0.1;
X = randn(nrCaract + 1, nrNeuroni) * 0.1;

%% METODA GRADIENT

iter = 1;
maxIter = 100000;
evolutieNorma = zeros(maxIter, 1);
evolutieFuncOb = zeros(maxIter, 1);
timp = zeros(maxIter, 1);
err = 1e-6;
pas = 1e-3;

start = tic;
while iter <= maxIter

    U = AExtins * X;
    Z = g(U);
    y = Z * x;
    r = y - eAntrenament;

    gradient_X = (1/nrExempleAntrenament) * AExtins' * ( (r * x') .* g_prim(U) );
    gradient_x = (1/nrExempleAntrenament) * Z' * r;

    timp(iter) = toc(start);

    evolutieFuncOb(iter) = (1 / (2*nrExempleAntrenament)) * norm(r)^2;

    norma = norm([gradient_X(:); gradient_x]);
    evolutieNorma(iter) = norma;

    if norma < err
        break;
    end

    X = X - pas * gradient_X;
    x = x - pas * gradient_x;

    iter = iter + 1;

end

figure('Name', 'METODA GRADIENT');
subplot(4, 1, 1);
semilogy(evolutieNorma);
title('Evolutia Normei Gradientului in Iteratii');
xlabel('Iteratie');
ylabel('Valoare');

subplot(4, 1, 2);
semilogy(evolutieFuncOb);
title('Evolutia Functiei Obiectiv in Iteratii');
xlabel('Iteratie');
ylabel('Valoare');

subplot(4, 1, 3);
semilogy(timp, evolutieNorma);
title('Evolutia Normei Gradientului in Timp');
xlabel('Timp');
ylabel('Valoare');

subplot(4, 1, 4);
semilogy(timp, evolutieFuncOb);
title('Evolutia Functiei Obiectiv in Timp');
xlabel('Timp');
ylabel('Valoare');

yTest = g(exempleTest * X) * x;
rTest = yTest - eTest;
medie_eTest = mean(eTest);

disp('Indicatori de performanta pentru Metoda Gradient:');
R2 = 1 - sum( rTest.^2 ) / sum( (eTest - medie_eTest).^2 )
MSE = (1/nrExempleTest) * sum( rTest.^2 )

%% METODA GRADIENT STOCHASTIC

iter = 1;
maxIter = 100000;
evolutieNorma = zeros(maxIter, 1);
evolutieFuncOb = zeros(maxIter, 1);
timp = zeros(maxIter, 1);
err = 1e-6;
pas = 1e-3;

dimensiuneBatch = 10;

start = tic;
while iter <= maxIter

    poz = randperm(nrExempleAntrenament, dimensiuneBatch);
    ABatch = AExtins(poz, :);
    eBatch = eAntrenament(poz);

    U = ABatch * X;
    Z = g(U);
    y = Z * x;
    r = y - eBatch;

    gradient_X = (1/dimensiuneBatch) * ABatch' * ( (r * x') .* g_prim(U) );
    gradient_x = (1/dimensiuneBatch) * Z' * r;

    timp(iter) = toc(start);

    evolutieFuncOb(iter) = (1 / (2*dimensiuneBatch)) * norm(r)^2;

    norma = norm([gradient_X(:); gradient_x]);
    evolutieNorma(iter) = norma;

    if norma < err
        break;
    end

    X = X - pas * gradient_X;
    x = x - pas * gradient_x;

    iter = iter + 1;

end

figure('Name', 'METODA GRADIENT STOCHASTIC');
subplot(4, 1, 1);
semilogy(evolutieNorma);
title('Evolutia Normei Gradientului in Iteratii');
xlabel('Iteratie');
ylabel('Valoare');

subplot(4, 1, 2);
semilogy(evolutieFuncOb);
title('Evolutia Functiei Obiectiv in Iteratii');
xlabel('Iteratie');
ylabel('Valoare');

subplot(4, 1, 3);
semilogy(timp, evolutieNorma);
title('Evolutia Normei Gradientului in Timp');
xlabel('Timp');
ylabel('Valoare');

subplot(4, 1, 4);
semilogy(timp, evolutieFuncOb);
title('Evolutia Functiei Obiectiv in Timp');
xlabel('Timp');
ylabel('Valoare');

yTest = g(exempleTest * X) * x;
rTest = yTest - eTest;
medie_eTest = mean(eTest);

disp('Indicatori de performanta pentru Metoda Gradient Stochastic:');
R2 = 1 - sum( rTest.^2 ) / sum( (eTest - medie_eTest).^2 )
MSE = (1/nrExempleTest) * sum( rTest.^2 )