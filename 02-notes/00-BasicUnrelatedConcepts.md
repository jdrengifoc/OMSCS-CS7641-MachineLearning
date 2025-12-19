
> [!EXAMPLE]+ Spaces ^00-spaces
> # Euclidean spaces
> Let $N \in \mathbb{N}$. The $n$**-dimensional Euclidean space**, is the pair $(\mathbb{R}, \langle \cdot, \cdot \rangle$, that is, the **vector space** of all $n$-tuples of real numbers equipped with the **Euclidean inner product** $\langle \cdot, \cdot \rangle: \mathbb{R}^n \times \mathbb{R}^n \rightarrow \mathbb{R}$.
> 
> ## Euclidean inner product
> Let $x,y \in \mathbb{R}^n$, the $n$-dimensional Euclidean inner product is defined as: $\langle x, y\rangle := \sum_{i=1}^n x_i y_i$.
> It induces 
> - **Euclidean norm**. $\| x \| = \sqrt{\langle x, x\rangle}$
> - **Euclidean distance.** $d(x, y) = \|x - y \|$
> ---
> # Vector space
> A **vector space** $V$ over a **field** $F$ is the triple $(V, +, \cdot)$ where:
> 1. $V$ is not empty set
> 2. $+: V \times V \rightarrow V$
> 3. $\cdot: F \times V \rightarrow V$
> 4. Satisfy some axioms.
> 
> ## Field 
> A **field** is a set, $F$, equipped with two binary operations, typically called **addition** (denoted by $+$) and **multiplication** (denoted by $\cdot$), that satisfy a set of specific axioms.
> 
> ---
> # Inner product
> Let $V$ a vector space  in $F$ (real or complex). An **inner product** is a map $\langle \cdot, \cdot \rangle: V \times V \rightarrow F$ satisfying, $\forall x, y, z \in V$ and $\alpha \in \mathbb{F}$:
> 1. **Linearity (in the first argument).** $\langle \alpha x + y, z \rangle = \alpha \langle x, z\rangle + \langle y, z\rangle$
> 2. **Symmetry.** $\langle x, y \rangle = \overline{\langle y, x \rangle}$
> 3. **Positive definiteness.** 
> 	1. $\langle x, x \rangle \geq 0$ 
> 	2. $\langle x, x \rangle = 0 \iff x = 0$
> 
> ## Remarks
> - Helps to determine angles and orthogonality
> - Induces a norm, which also induces a distance.
> - Requires a vector space (e.g., function spaces $L^2$ and Hilbert spaces).
> - Is part of a **inner product space**.
> ---
> # Norm
>  Let $V$ a vector space in $F$ (usually an ordered field). A **norm** is a map $\| \cdot \| : V \times V \rightarrow [0, \infty)$ such that, $\forall x, y \in V$ and $\alpha \in F$:
>  1. **Positivity.** 
> 	 - $\| x \| \geq 0$
> 	 - $\| x \| = 0 \iff x = 0$
> 2. **Absolute homogeneity.** $\| \alpha x \| = |\alpha|\| x \|$
> 3. **Triangle inequality.** $\|  x + y\| \leq \| x \| + \| y \|$
> 
> ## Remarks
> - Helps to determine the length of an object.
> - Induces a distance
> - Not every norm comes from an inner product.
> - Is part of a **normed space**.
> ---
> # Distance (metric)
> Let $X$ be a set. A **distance** is a map $d: X \times X \rightarrow [0, \infty)$ satisfying, $\forall x, y, z \in X$:
> 1. **Non-negativity.** $d(x, y) \geq 0$
> 2. **Identity of indiscernibles.** $d(x, y) = 0 \iff x = y$
> 3. **Symmetry.** $d(x, y) = d(y, x)$
> 4. **Triangular inequality.** $d(x, y) \leq d(x, z) + d(z, y)$.
> 
> ## Remarks
> - Introduces the separation between objects in a set.
> - Not every distance come from a norm.
> - Is part of a metric space.
> ---
> # Summary
> ```mermaid
> flowchart TD
> VS[Vector space]
> NS[Normed space]
> IPS[Inner product space]
> MS[Metric space]
> S[Set]
> IP((Inner<br/>product))
> N((Norm))
> M((Metric))
> IPS -.-> |induces| N
> NS -.-> |induces| M
> IP --> IPS
> N --> NS
> M --> MS
> VS --> IPS
> VS --> NS
> VS -.-> S
> S --> MS
>  ```
> > | Concept | Underlying Set | Field Needed | Linear Structure | Geometry |
> > | :--- | :--- | :--- | :--- | :--- |
> > | Vector Space | Set | Any field ($F$) | Addition + Scalar Mult. | None |
> > | Normed Space | Vector Space | Usually ordered ($F=\mathbb{R}$ or $\mathbb{C}$) | Length (Norm) | Size |
> > | Inner Product Space | Vector Space | $\mathbb{R}$ or $\mathbb{C}$ | Full Linear + Length (Norm) | Angles, Orthogonality |
> > | Metric Space | Set | None | None | Distance |

> [!EXAMPLE]+ Probability Space ^00-propability-space
> 
> # $\sigma$-algebra
> Let $X$ a set, then a the subset $\mathcal{A} \subseteq \mathcal{P}(X)$ is a $\sigma$-algebra if satisfy the following properties:
> 1. $X \in \mathcal{A}$;
> 2. $\mathcal{A}$ is **closed under complementation**. $A \in \mathcal{A} \implies A^c \in \mathcal{A}$;
> 3. $\mathcal{A}$ is **closed under countable unions**. $A_1, A_2,... \in \mathcal{A} \implies \bigcup_{n \in N} A_n \in \mathcal{A}$.
> 
> ## Corollaries
> 1. $\mathcal{A}$ is **closed under countable intersections**. Applying D'Morgan.
> 2. $\varnothing \in \mathcal{A}$.
> 3. The smallest $\sigma$-algebra is $\{X, \varnothing\}$.
> 4. The largest $\sigma$-algebra is $\mathcal{P}(X)$.
> **Note:** An element in a $\sigma$-algebra is called a **measurable set**.
> 
> ---
> 
> # Measure
> Let $X$ a set, $\mathcal{A}$ its $\sigma$-algebra. A **set function** $\mu: \mathcal{A} \rightarrow \mathbb{R}$ is a **measure** if satisfies the following properties:
> 1. $\mu(\varnothing) = 0$;
> 2. **Non-negativity.** $\forall A \in mathcal{A} \implies \mu(A) \geq 0$.
> 3. **Countable additivity.** For all countable collection $\{A_n\}_{n \in N}$ of **pairwise disjoint sets** in $\mathcal{A}$, that is, $\forall i, j \in N \implies A_i \cap A_j = \varnothing$:
>    $$\mu \left( \bigcup_{n \in N} A_n \right) = \sum_{n \in N} \mu(A_n)$$
> 
> # Probability Measure
> A **measure** $P$ for the set $\Omega$ is a **probability measure** if $P(\Omega) = 1$.
> - Its range is $[0,1]$
>
> ---
> 
> # Measure Space
> A **measure space** is a triple $(X, \mathcal{A}, \mu)$ where:
> 1. $X$ is a set;
> 2. $\mathcal{A}$ is a $\sigma$-algebra on the set $X$;
> 3. $\mu$ is  **measure** on the **measurable space**, or **Borel space** $(X, \mathcal{A})$.
> 
> ---
> 
> # Probability space
> A **probability space** is a triple $(\Omega, \mathcal{F}, P)$ where:
> 1. $\Omega$ is a the **sample space**, that is, the set of all possible **outcomes** of a **random process**;
> 2. $\mathcal{F}$ is an **event space**, that is, a $\sigma$-algebra on the set $\Omega$ understood as the set of events;
> 3. $P: \mathcal{F} \rightarrow [0,1]$ is **probability measure** on the **measurable space** $(\Omega, \mathcal{F})$.

> [!EXAMPLE]+ Sets ^00-sets
> # Basic concepts
> The **power set** of a set $X$, denoted by $\mathcal{P}(X)$ is the set of all subsets of $X$ including the **empty set** $\varnothing$ and $X$.

> [!EXAMPLE]+ Expected Value ^00-expected-value
> # Basic concepts
> ## Random variable
>  Let $(\Omega, \mathcal{F}, P)$ a **probability space**. A **random variable** is a **measurable function**  $X: \Omega \rightarrow \mathbb{R}$. Formally, for every **borel set** $B \in \mathcal{B}(\mathbb{R})$, the preimage $X^{-1}(B) \in \mathcal{F}$.
>  - Different to a **measure**, which assigns a importance to each event, while a **measurable function** assigns a label to outcomes of interest.
>  ## Lebesgue integral
>  
>  ### 1. Simple functions
>  Let $(\Omega, \mathcal{F}, P)$ a **probability space**. A function $\phi: \omega \rightarrow \mathbb{R}$.is called a **Simple Function** if it is a **measurable function** with a **finite range**.
> 
> **Theorem.** Every simple function can be uniquely represented as a linear combination of **indicator functions** such as:
> $$\phi(\omega) = \sum_{i=1}^n a_i \mathbb{1}_{A_i}(\omega),$$
> where:
> 1. $a_i$ are distinct real values in the range
> 2. $A_i = \{\omega \in \Omega: \phi(\omega) = a_i\}$ are the **preimages** of $a_i$
> 	1. The sets $A_i$ form a **partition** on $\Omega$ (Disjoint and their union is $\Omega$)
> 	2. $A_i \in \mathcal{F}$
> 
> **Definition.** The **Lebesgue integral for a simple function** $\phi$ with respect to a **probability measure** $P$ is:
> $$
> \int_{\Omega} \phi dP = \sum_{i=1}^n a_i P(A_i)
> $$
> - Identical to the discrete expected value.
> 
> ### 2. Non-negative measurable functions
> For any non-negative random variable $X \geq 0$, we define the integral by approximating $X$ from below using **simple functions**. Let $S_X$ the set of all simple functions $s$ such that $0 \leq s \leq X$.
> 
> $$
> \int_{\Omega} X dP = \sup \left\{ \int_\Omega s dP: s \in S_X\right\}
> $$
> 
> ### 3. General measurable functions
> For a random variable $X$, we decompose it into its positive and negative parts, such as, $X = X^+ - X^-$ where: $X^+ = \max(X,0)$ and $X^- = \max(-X, 0)$. Hence,
> $$
> \int_{\Omega} X dP = \int_{\Omega} X^+ dP - \int_{\Omega} X^- dP
> $$
>



> [!ERROR]+ Lebesgue Integral INCOMPLETE ^00-lebesgue-integral
> 
> The Lebesgue integral generalizes the Riemann integral by changing how the area under the function $f(x)$ is calculated.
> https://www.youtube.com/watch?v=gHUZFXvy4yE
> https://www.youtube.com/watch?v=PGPZ0P1PJfw
> 
> ### 🧠 Key Difference: Slicing the Range
> 
> The method partitions the **range (Y-axis)** of the function, rather than the domain (X-axis). It sums the size of the sets of inputs that map to each output value, weighted by those output values. 
> 
> > ---
> 
> ### 📐 Formal Representation
> 
> The integral of a **non-negative function** $f$ over a **measure space** $(X, \mathcal{A}, \mu)$ is defined as the supremum of integrals of **simple functions**:
> 
> $$\int_X f \, d\mu = \sup \left\{ \int_X s \, d\mu \mid s \text{ is simple, } 0 \le s \leq f \right\}$$
> 
> **Where:**
> * $d\mu$ is the **measure** (length, area, probability) element.
> * $\mu$ must satisfy $\sigma$-additivity.
> 
> ### ✅ Advantages
> 
> * **Integrates More Functions:** It can integrate functions that are too pathological for the Riemann integral, such as the **Dirichlet function**
> * **Better Limit Properties:** It satisfies crucial theorems for interchanging limits and integration.

> [!ERROR]+ Kernels INCOMPLETE ^00-kernels
> # KNN
> function of distance that is used to determine the weight of each training example. In other words, the kernel function is the function $K$ such that $w_i$ = $K(d(x_i, x))$.

