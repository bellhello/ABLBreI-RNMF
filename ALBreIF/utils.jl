function soft_thresholding(A::AbstractArray{T}, s::Real) where {T}
    B = copy(A)
    n = length(B)
    @inbounds for i = 1:n
        x = sign(B[i]) * max(abs(B[i]) - s, 0)
        B[i] = x
    end
    return B
end

function projectnn!(A::AbstractArray{T}) where {T}
    # project back all entries to non-negative domain
    @inbounds for i = 1:length(A)
        if A[i] < zero(T)
            A[i] = zero(T)
        end
    end
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