import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- Page Config ---
st.set_page_config(
    page_title="Multirate & Filter Explorer",
    page_icon="🌊",
    layout="wide"
)

# --- CSS for Professional Styling ---
st.markdown("""
<style>
    .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #bdc6f2;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #ff4b4b;
    }
    .metric-box {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("### 🌊 Filter Design & Noble Identities")
# st.title("🌊 Multirate Signal Processing & Filter Design")
# st.markdown("""
# Explore the core concepts of **FIR Filter Design**, **Noble Identities**, and **Polyphase Decomposition**.
# """)

# --- Create the Tabs ---
tab1, tab2, tab3 = st.tabs([
    "1️⃣ FIR Design (Remez)", 
    "2️⃣ Noble Identities", 
    "3️⃣ Polyphase Efficiency"
])

# ==============================================================================
# TAB 1: FIR FILTER DESIGN (Remez/Parks-McLellan)
# ==============================================================================
with tab1:
    # st.header("1. Filter Design (Parks-McLellan)")
    st.markdown("Design optimal Equi-ripple FIR filters using the `remez` algorithm.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # st.subheader("Specs")
        N = st.slider("Filter Order (N)", 8, 128, 32, step=2, help="Length of impulse response")
        
        # Band Edges (Normalized 0 to 0.5)
        st.markdown("**Band Edges ($f/f_s$)**")
        pass_edge = st.slider("Passband End", 0.01, 0.45, 0.1, 0.01)
        stop_edge = st.slider("Stopband Start", pass_edge + 0.01, 0.49, 0.2, 0.01)
        
        # Weights
        st.markdown("**Weights**")
        w_pass = st.number_input("Passband Weight", 1.0, 100.0, 1.0)
        w_stop = st.number_input("Stopband Weight", 1.0, 1000.0, 100.0)
        
    with col2:
        # --- Calculation ---
        # Bands for Remez: [0, pass_edge, stop_edge, 0.5]
        bands = [0, pass_edge, stop_edge, 0.5]
        desired = [1, 0] # Gain 1 in pass, 0 in stop
        weights = [w_pass, w_stop]
        
        try:
            h = signal.remez(N, bands, desired, weight=weights, fs=1.0)
            
            # Frequency Response
            w, H = signal.freqz(h, worN=2048)
            freq_norm = w / (2 * np.pi) # 0 to 0.5
            H_db = 20 * np.log10(np.abs(H) + 1e-12)
            
            # --- Plotting ---
            fig1, (ax1_freq, ax1_time) = plt.subplots(2, 1, figsize=(10, 8))
            fig1.patch.set_alpha(0)
            
            # 1. Frequency Response
            ax1_freq.plot(freq_norm, H_db, 'b-', linewidth=1.5)
            # Draw Ideal specs
            ax1_freq.fill_between([0, pass_edge], -100, 10, color='green', alpha=0.1, label="Passband")
            ax1_freq.fill_between([stop_edge, 0.5], -100, 10, color='red', alpha=0.1, label="Stopband")
            
            ax1_freq.set_title("Frequency Response (Magnitude)", loc='left')
            ax1_freq.set_ylabel("Magnitude (dB)")
            ax1_freq.set_xlabel("Normalized Frequency ($f/f_s$)")
            ax1_freq.set_ylim(-80, 5)
            ax1_freq.set_xlim(0, 0.5)
            ax1_freq.grid(True, alpha=0.3)
            ax1_freq.legend(loc="upper right")
            
            # 2. Impulse Response
            ax1_time.stem(np.arange(len(h)), h, basefmt=" ", linefmt='k-', markerfmt='bo')
            ax1_time.set_title(f"Impulse Response $h[n]$ (N={N})", loc='left')
            ax1_time.set_xlabel("Sample $n$")
            ax1_time.grid(True, alpha=0.3)
            
            st.pyplot(fig1)
            
            # Check Linear Phase (Symmetry)
            is_symmetric = np.allclose(h, h[::-1], atol=1e-5)
            if is_symmetric:
                st.success("✅ Filter has Linear Phase (Symmetric Impulse Response)")
            else:
                st.warning("⚠️ Filter is NOT Linear Phase")
                
        except Exception as e:
            st.error(f"Design Error: {e}. Try adjusting band edges.")

# ==============================================================================
# TAB 2: NOBLE IDENTITIES
# ==============================================================================
with tab2:
    # st.header("2. Noble Identities (Verification)")
    # st.markdown(r"""
    # Verify the identity: **$\downarrow N \circ H(z) \equiv H(z^N) \circ \downarrow N$**
    
    # *Left Side:* Downsample first, then filter.  
    # *Right Side:* Filter with upsampled $H(z^N)$ first, then downsample.
    # """)
    st.markdown(r"""
    Verify the identity: **$\downarrow N \circ H(z) \equiv H(z^N) \circ \downarrow N$**
    """)
    
    col_n1, col_n2 = st.columns([1, 3])
    
    with col_n1:
        M = st.selectbox("Downsample Factor (M)", [2, 3, 4])
        signal_type = st.selectbox("Input Signal", ["Step", "Ramp", "Random"])
        
    with col_n2:
        # Setup
        h_simple = np.array([1, 2, 3, 2, 1]) # Simple filter
        
        # Input Signal x
        L = 20
        if signal_type == "Step":
            x = np.ones(L)
        elif signal_type == "Ramp":
            # x = np.arange(L)
            x = np.arange(L, dtype=float) # Force float
        else:
            np.random.seed(42)
            # x = np.random.randint(0, 10, L)
            x = np.random.randint(0, 10, L).astype(float) # Force float
            
        # --- Left Side: Downsample -> Filter ---
        x_down = x[::M]
        y_left = signal.lfilter(h_simple, 1, x_down)
        
        # --- Right Side: Filter(Up) -> Downsample ---
        # Upsample filter (insert M-1 zeros)
        h_up = np.zeros(len(h_simple) * M - (M-1))
        h_up[::M] = h_simple
        
        y_temp = signal.lfilter(h_up, 1, x)
        y_right = y_temp[::M]
        
        # --- Visualization ---
        fig2, ax2 = plt.subplots(3, 1, figsize=(10, 8))
        fig2.patch.set_alpha(0)
        
        # Plot Input
        ax2[0].stem(x, basefmt=" ", linefmt='k-', label="Input x")
        ax2[0].set_title("Input Signal", fontsize=12)
        
        # Plot Left Result
        # ax2[1].stem(y_left, basefmt=" ", linefmt='C0-', markerfmt='C0o', label="Left Side")
        ax2[1].stem(y_left, basefmt=" ", linefmt='b-.', markerfmt='bo', label="Left Side")
        # ax2[1].set_title(f"Left: Downsample by {M} → Filter H(z)", loc='left', fontsize=10, color='blue')
        ax2[1].set_title(f"Downsample by {M} → Filter H(z)", fontsize=12)
        
        # Plot Right Result
        # Add small offset to x-axis to see overlap if they match
        # ax2[2].stem(y_right, basefmt=" ", linefmt='C1--', markerfmt='C1x', label="Right Side")
        ax2[2].stem(y_right, basefmt=" ", linefmt='r--', markerfmt='rx', label="Right Side")
        # ax2[2].set_title(f"Right: Filter H(z^{M}) → Downsample by {M}", loc='left', fontsize=10, color='orange')
        ax2[2].set_title(f"Filter H($z^{M}$) → Downsample by {M}", fontsize=12)
        
        for ax in ax2:
            ax.grid(True, alpha=0.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        fig2.tight_layout()
        fig2.subplots_adjust(hspace=0.5)
        st.pyplot(fig2)
        
        # Verification
        if np.allclose(y_left, y_right):
            st.success(f"✅ Identity Verified! The outputs are identical.")
        else:
            st.error("❌ Mismatch found (check boundary conditions).")

# ==============================================================================
# TAB 3: POLYPHASE EFFICIENCY
# ==============================================================================
with tab3:
    # st.header("3. Polyphase Decomposition Efficiency")
    st.markdown("Visualize how a filter is split into $M$ polyphase components to save computation.")
    with st.container(border=True):
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            M_poly = st.slider("Decimation Factor M", 2, 8, 4)
        with col_p2:
            N_poly = st.slider("Filter Length (Taps)", 16, 128, 32, step=M_poly)
            
    
    # col_p1, col_p2 = st.columns([1, 2])
    
    # with col_p1:
    #     M_poly = st.slider("Decimation Factor M", 2, 8, 4)
    #     N_poly = st.slider("Filter Length (Taps)", 16, 128, 32, step=M_poly)
        
    # with col_p2:
    # Create a dummy filter
    h_poly = np.arange(N_poly) + 1 # 1, 2, 3...
        
    # Decompose
    components = []
    for i in range(M_poly):
        # Take every Mth sample starting at i
        sub_filter = h_poly[i::M_poly]
        components.append(sub_filter)
            
    # --- Visualize Components ---
    # st.subheader("Polyphase Components $H_i(z)$")
    st.markdown("""Polyphase Components $H_i(z)$""")
        
    # We plot the original and color-code the components
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    fig3.patch.set_alpha(0)
        
    # Plot original in gray shadow
    ax3.stem(h_poly, linefmt='gray', markerfmt='k.', basefmt=" ", label="Original H(z)")
        
    # Plot first 2 components colored to show the pattern
    # Component 0
    idx0 = np.arange(0, N_poly, M_poly)
    ax3.stem(idx0, h_poly[idx0], linefmt='r-', markerfmt='ro', basefmt=" ", label=f"H0 (Phase 0)")
        
    # Component 1
    idx1 = np.arange(1, N_poly, M_poly)
    ax3.stem(idx1, h_poly[idx1], linefmt='b-', markerfmt='bx', basefmt=" ", label=f"H1 (Phase 1)")
        
    ax3.set_title(f"Decomposing H(z) (Length {N_poly}) into {M_poly} Filters", loc='left')
    ax3.legend()
    ax3.grid(True, alpha=0.2)
    st.pyplot(fig3)
        
    # --- Cost Calculation ---
    # st.subheader("💡 Computational Cost Analysis")
    st.markdown("""📈 Computational Cost Analysis""")
        
    # Let's assume input signal length L_sig
    L_sig = 10000 
        
    # Direct: Filter (L_sig * N_poly) then drop M-1 samples
    ops_direct = L_sig * N_poly
        
    # Polyphase: Commutator splits signal (no cost) -> M filters run at L_sig/M rate
    # Length of sub-filters is N_poly/M
    # Each sub-filter processes (L_sig/M) samples
    # Ops per sub-filter = (L_sig/M) * (N_poly/M)
    # Total ops = M * [(L_sig/M) * (N_poly/M)] = (L_sig * N_poly) / M
        
    ops_poly = ops_direct / M_poly
        
    c1, c2, c3 = st.columns(3)
    c1.metric("Direct Ops", f"{ops_direct:,}")
    c2.metric("Polyphase Ops", f"{int(ops_poly):,}")
    c3.metric("Speedup Factor", f"{M_poly}x", delta="Efficient!")
        
    # --- Educational Expander ---
    with st.expander("🚀 Why is it faster?"):
        st.markdown(r"""Instead of calculating convolution for *every* sample and then throwing away $M-1$ of them (Direct Downsampling), 
        we only calculate the convolution for the samples we actually keep!
        """)
