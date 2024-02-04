# Numerical utilities to support implementation

using LinearAlgebra.BLAS: nrm2
using LinearAlgebra.LAPACK: potrf!, potri!, potrs!

function hard_thresholding(A::AbstractArray{T}, s::Real) where {T}
    B = copy(A)
    L = length(A)
    S = floor(s * L)
    B = sort(B, rev=true)
    S = convert(Int64, S)
    hs = B[S]
    l = length(A)
    @inbounds for i = 1:l
        if A[i] <= hs
            A[i] = zero(T)
        end
    end
end

function soft_thresholding(A::AbstractArray{T}, s::Real) where {T}
    B = copy(A)
    n = length(B)
    @inbounds for i = 1:n
        x = sign(B[i]) * max(abs(B[i]) - s, 0)
        B[i] = x
    end
    return B
end

function printf_mat(x::AbstractMatrix)
    m, n = size(x)
    @inbounds for i = 1:m
        for j = 1:n
            @printf("%8.4f ", x[i, j])
        end
        println()
    end
end

function adddiag!(A::Matrix, a::Number)
    m, n = size(A)
    m == n || error("A must be square.")
    if a != 0.0
        for i = 1:m
            @inbounds A[i, i] += a
        end
    end
    return A
end

normalize1!(a) = rmul!(a, 1 / sum(a))

function normalize1_cols!(a)
    for j = 1:size(a, 2)
        normalize1!(view(a, :, j))
    end
end

function projectnn!(A::AbstractArray{T}) where {T}
    # project back all entries to non-negative domain
    @inbounds for i = 1:length(A)
        if A[i] < zero(T)
            A[i] = zero(T)
        end
    end
end

function posneg!(A::AbstractArray{T},
    Ap::AbstractArray{T}, An::AbstractArray{T}) where {T}
    # decompose A into positive part Ap and negative part An
    # s.t. A = Ap - An

    n = length(A)
    length(Ap) == length(An) == n || error("Input dimensions mismatch.")

    @inbounds for i = 1:n
        ai = A[i]
        if ai >= zero(T)
            Ap[i] = ai
            An[i] = zero(T)
        else
            Ap[i] = zero(T)
            An[i] = -ai
        end
    end
end

function pdsolve!(A, x, uplo::Char='U')
    # A must be positive definite
    # x <- inv(A) * x
    # both A and x will be overriden

    potrf!(uplo, A)
    potrs!(uplo, A, x)
end

function pdrsolve!(A, B, x, uplo::Char='U')
    # B must be positive definite
    # x <- A * inv(B)
    # both B and x will be overriden

    # inverse B in place
    potrf!(uplo, B)
    potri!(uplo, B)
    copytri!(B, uplo)

    # x <- A * B (the inversed one)
    mul!(x, A, B)
end

function find_positive_root(g, a, b, tol)
    fa = g(a)
    fb = g(b)
    
    if fa * fb > 0
        error("The function g does not have a unique positive root in the interval [$a, $b].")
    end
    
    while b - a > tol
        c = (a + b) / 2
        fc = g(c)
        
        if fc == 0
            return c
        elseif fc * fa < 0
            b = c
            fb = fc
        else
            a = c
            fa = fc
        end
    end
    
    return (a + b) / 2
end