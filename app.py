import streamlit as st
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

st.set_page_config(page_title="Dashboard",page_icon="⚛",layout="wide")
#https://discuss.streamlit.io/t/using-custom-fonts/14005 
#t = st.radio("Toggle to see font change", [True, False])

if True:
    st.markdown(
    """
<style>
            @import url('https://fonts.googleapis.com/css2?family=Inter&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter'; 
            }
            section[data-testid="stSidebar"] {
            width: 460px !important; # Set the width to your desired value
            }
</style>
    """,
        unsafe_allow_html=True,
    )
st.sidebar.image("NobleAI_Logo_Reactor_Blk-Blu.png",caption="NobleAI <> Internal")


# Function to visualize molecules
def visualize_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    #img = Draw.MolToImage(mol)
    #st.image(img, caption=smiles, use_column_width=False)
    d1 = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(300,200)
    d1.DrawMolecule(mol)
    d1.FinishDrawing()
    svg1 = d1.GetDrawingText().replace('svg:','')
    st.image(svg1)

# Function to calculate similarity scores
def calculate_similarity(user_smiles, smiles_list):
    c_user_smiles = []
    c_smiles = []

    try:
        cs_user = Chem.CanonSmiles(user_smiles)
        c_user_smiles.append(cs_user)
    except:
        st.warning(f'Invalid user input SMILES: {user_smiles}')
        return None

    for ds in smiles_list:
        try:
            cs = Chem.CanonSmiles(ds)
            c_smiles.append(cs)
        except:
            st.warning(f'Invalid SMILES: {ds}')

    ms_user = Chem.MolFromSmiles(c_user_smiles[0])
    ms = [Chem.MolFromSmiles(x) for x in c_smiles]

    fps_user = FingerprintMols.FingerprintMol(ms_user)
    fps = [FingerprintMols.FingerprintMol(x) for x in ms]

    qu, ta, sim = [], [], []

    for n in range(len(fps)):
        s = DataStructs.FingerprintSimilarity(fps_user, fps[n])
        qu.append(user_smiles)
        ta.append(c_smiles[n])
        sim.append(s)

    d = {'query': qu, 'target': ta, 'Similarity': sim}
    df_result = pd.DataFrame(data=d)
    df_result = df_result.sort_values('Similarity', ascending=False)
    
    return df_result

# Streamlit app
def main():
    st.title("SBAI and Chemical Similarity for Colorants")

    # Load data
    df = pd.read_csv('eu_annex_iv_smiles.csv')
    df_smiles = df['SMILES']


    # Display original DataFrame
    # with st.sidebar:
    st.subheader("EU Annex IV:")
    st.write(df)

    # show internal diversity
    ms = [Chem.MolFromSmiles(x) for x in df["SMILES"]]
    #fpgen = AllChem.GetRDKitFPGenerator()
    #fps = [ fpgen.GetSparseCountFingerprint(m) for m in ms ]
    
    fps0 = [ AllChem.GetMorganFingerprintAsBitVect(m,radius=2,nBits=512) for m in ms]
    fps = []
    for fp in fps0:
        fingerprint_array = np.zeros((1,), dtype=int)  # Create an empty array to hold the fingerprint
        DataStructs.ConvertToNumpyArray(fp, fingerprint_array) 
        fps.append(fingerprint_array)
    fps = np.array(fps)

    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(fps)

    df2 = df.copy()
    df2["dim1"] = X_2d[:,0]
    df2["dim2"] = X_2d[:,1]
    fig = px.scatter(df2,x="dim1",y="dim2",color="Color", color_discrete_sequence = df['Color'].unique(),
                    #  color_continuous_scale=px.colors.sequential.Viridis,
                        hover_name="Colour index Number",
                        # hover_data=["SMILES"],
                    #  labels={"log10ODT":"log10 ODT"}
                        )

    fig.update_layout(
                width=425,
                height=425,
                font_family='Inter',
                font_size=14
                # title='Test'
            )
    fig.update_traces(marker=dict(size=12,
                                        line=dict(width=2,
                                                    color='DarkSlateGrey')),
                            selector=dict(mode='markers'))
    st.plotly_chart(fig, theme="streamlit", use_container_width=False)
        

    # User input
    user_smiles = st.text_input("Enter a SMILES string for comparison:")

    if user_smiles:
        # Calculate similarity and display result
        df_result = calculate_similarity(user_smiles, df_smiles)
        df_result['SMILES'] = df['SMILES']
        df_result['Reference Number'] = df['Reference Number']
        df_result['Colour index Number'] = df['Colour index Number']

        c1,c2 = st.columns(2)
        # Visualize user input molecule
        with c1:
            st.subheader("Visualize User Input Molecule:")
            visualize_molecule(user_smiles)

        if df_result is not None:
            with c2:
                st.subheader("Similarity Comparison Result:")
                fig = px.scatter(df_result, 'Similarity', 'Reference Number', hover_name='Colour index Number')
                fig.update_layout(width=450, height=450)

                fig.update_layout(
                    width=450,
                    height=450,
                    font_family='Inter',
                    font_size=14
                    # title='Test'
                )

                fig.update_traces(marker=dict(size=16,
                                            line=dict(width=2,
                                                        color='DarkSlateGrey')),
                                selector=dict(mode='markers'))
                # fig.update_traces(marker={'size': 14, 'line_width':2, 'line_color':'DarkSlateGrey'})


                st.plotly_chart(fig, theme="streamlit", use_container_width=False)

            # Display result DataFrame
            with st.expander("raw results"):
                st.write(df_result)

        # Save result as CSV
        df_result.to_csv('result.csv', index=False, sep=',')

if __name__ == "__main__":
    main()
