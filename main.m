clc, clear
n = 70;                              % valoarea de redimensionare a img.
A_bar = imread("pisica.jpg");         % citirea img
A_bar = rgb2gray(A_bar);             % convertire in alb-negru
A_bar =im2double(A_bar);             % convertirea in valori double
A_bar = imresize(A_bar,[n n]);       % redimensionarea  pozei  
figure(1)
imshow(A_bar);                           % afisarea pozei initiale 
title("Imaginea originala")

nrintraricunoscute=3000;             % setam un numar de intrari cunoscute 
rPerm = randperm(n*n);                          %generarea random a indicilor pentru intrarile cunoscuti
omega = sort(rPerm(1 : nrintraricunoscute));    %intrarile care se cunosc
A = nan(n); A(omega) = A_bar(omega); 
figure(2)
title("Imagine cu pixeli lipsa")
imshow(A) 

 cvx_begin 
 variable B(n,n)
 minimize(norm_nuc(B))
 subject to:
 B(omega) == A(omega)
 cvx_end
 figure(3);
 imshow(B);
 title('CVX');


% Parametrii pentru gradientul accelerat
maxiter = 1000;
eps = 1e-3;
alpha = 0.1;  % rata de învățare
beta = 0.9;    % factor de impuls

% Inițializare
A_initial = randn(n, n);
A_initial(omega) = A(omega);
Y = A_initial;
X_old = A_initial;
oprire = 1;
iter = 0;
vector_oprire = [];
timp = zeros(1, maxiter); % vector pentru stocarea timpului

% Măsurarea timpului de execuție la fiecare iterație
tic;

while oprire >= eps && iter < maxiter
    vector_oprire = [vector_oprire, oprire];
    grad = Y - A_bar;  % Calculul gradientului
    
    % Actualizarea X_new utilizând gradientul
    X_new = Y - alpha * grad;
    X_new(omega) = A(omega);  % Impunerea constrângerii
    
    % Calculul pasului de impuls
    Y = X_new + beta * (X_new - X_old);
    
    oprire = norm(X_new - X_old);
    X_old = X_new;
    iter = iter + 1;
    
    % Măsurarea timpului la fiecare iterație
    timp(iter) = toc;
end

% Afișarea imaginii recuperate
figure;
imshow(X_new);
title('Imagine recuperată MG Accelerat');

% Trasarea graficului cu evoluția criteriului de oprire în funcție de numărul de iterații
figure;
subplot(2, 1, 1);
semilogy(1:iter, vector_oprire, 'LineWidth', 2);
title('Evoluția criteriului de oprire în funcție de numărul de iterații MG accelerat');
xlabel('Numărul de iterații');
ylabel('Criteriul de oprire');

% Trasarea graficului cu evoluția criteriului de oprire în funcție de timp
subplot(2, 1, 2);
semilogy(timp(1:iter), vector_oprire, 'LineWidth', 2);
title('Evoluția criteriului de oprire în funcție de timp MG accelerat');
xlabel('Timp (s)');
ylabel('Criteriul de oprire');


% Parametrii pentru gradientul stocastic
alpha_sgd = 0.1;  % rata de învățare pentru SGD (mai mică pentru a asigura convergența)
maxiter_sgd = 1000;
eps_sgd = 1e-3;
batch_size = 200;   % dimensiunea subsetului aleator de date pentru SGD
X_sgd = A_initial; % Începem de la imaginea inițială

% Buclă pentru SGD
oprire_sgd = 1;
iter_sgd = 0;
vector_oprire_sgd = []; % vector pentru stocarea criteriului de oprire în timpul iterațiilor
timp_sgd = []; % vector pentru stocarea timpului

% Măsurarea timpului de execuție la fiecare iterație pentru SGD
tic;

while oprire_sgd >= eps_sgd && iter_sgd < maxiter_sgd
    % Alegerea unui subset aleator de date (batch) pentru gradientul stocastic
    idx = randperm(length(omega), batch_size);
    xi_batch = A_bar(omega(idx));

    % Calculul gradientului parțial folosind subsetul de date selectat
    grad_sgd = X_sgd - repmat(mean(xi_batch, 2), [1, size(X_sgd, 2)]);

    % Actualizarea matricei folosind gradientul parțial estimat
    X_new_sgd = X_sgd - alpha_sgd * grad_sgd;
    X_new_sgd(omega) = A(omega);  % Impunerea constrângerii

    oprire_sgd = norm(X_new_sgd - X_sgd);
    X_sgd = X_new_sgd;
    iter_sgd = iter_sgd + 1;

    % Stocarea criteriului de oprire în timpul iterațiilor
    vector_oprire_sgd = [vector_oprire_sgd, oprire_sgd];

    % Măsurarea timpului la fiecare iterație pentru SGD
    timp_sgd(iter_sgd) = toc;
end

% Afișarea imaginii recuperate folosind metoda de gradient stocastic (SGD)
figure;
imshow(X_sgd);
title('Imagine recuperată folosind metoda de gradient stocastic (SGD)');

% Trasarea graficului cu evoluția criteriului de oprire în funcție de numărul de iterații pentru SGD
figure;
subplot(2, 1, 1);
semilogy(1:iter_sgd, vector_oprire_sgd, 'LineWidth', 2);
title('Evoluția criteriului de oprire în funcție de numărul de iterații pentru SGD');
xlabel('Numărul de iterații');
ylabel('Criteriul de oprire');

% Trasarea graficului cu evoluția criteriului de oprire în funcție de timp pentru SGD
subplot(2, 1, 2);
semilogy(timp_sgd(1:iter_sgd), vector_oprire_sgd, 'LineWidth', 2);
title('Evoluția criteriului de oprire în funcție de timp pentru SGD');
xlabel('Timp (s)');
ylabel('Criteriul de oprire');
