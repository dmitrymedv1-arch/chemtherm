""")

# Основная область
if load_button and data_input:
# Загрузка данных
with st.spinner('Loading data...'):
df_raw = parse_uploaded_data(data_input)

if df_raw is not None and not df_raw.empty:
    st.success(f'✅ Data loaded successfully! {len(df_raw)} samples found.')
    
    # Очистка данных
    df_clean = clean_data(df_raw)
    
    # Расчет дескрипторов
    with st.spinner('Calculating descriptors...'):
        engine = DescriptorEngine(df_clean)
        desc_df = engine.calculate_all()
    
    # Объединение с исходными данными
    full_df = pd.concat([df_clean, desc_df], axis=1)
    
    # Сохранение в session_state
    st.session_state['df'] = full_df
    st.session_state['df_raw'] = df_raw
    st.session_state['desc_df'] = desc_df
    
    # Отображение вкладок
    display_tabs(full_df, desc_df, df_raw)
else:
    st.error('❌ Failed to parse data. Please check the format.')

elif 'df' in st.session_state:
# Использование загруженных данных из session_state
display_tabs(st.session_state['df'], 
        st.session_state['desc_df'],
        st.session_state['df_raw'])
else:
st.info('👈 Please paste your data in the sidebar and click "Load Data" to start analysis.')

def display_tabs(df, desc_df, df_raw):
"""
Отображение вкладок с анализом.
"""
# Создание вкладок
tabs = st.tabs([
'📊 Data Overview',
'🔬 Descriptors',
'📈 Correlations',
'🧬 PCA & Clustering',
'📊 Visualizations',
'💾 Export'
])

# Вкладка 1: Обзор данных
with tabs[0]:
st.header('📊 Data Overview')

col1, col2, col3 = st.columns(3)
with col1:
st.metric('Total Samples', len(df))
with col2:
st.metric('Total Features', len(df.columns))
with col3:
missing = df.isnull().sum().sum()
st.metric('Missing Values', missing)

st.subheader('Sample Data')
st.dataframe(df.head(10))

st.subheader('Data Statistics')
st.dataframe(df.describe())

st.subheader('Missing Values Matrix')
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
ax.set_title('Missing Values', fontsize=12, fontweight='bold')
st.pyplot(fig)
plt.close()

# Вкладка 2: Дескрипторы
with tabs[1]:
st.header('🔬 Descriptor Engine')

st.subheader('Calculated Descriptors')
st.dataframe(desc_df)

# Распределение дескрипторов
st.subheader('Descriptor Distributions')

# Выбор дескрипторов для отображения
desc_cols = desc_df.select_dtypes(include=[np.number]).columns.tolist()
selected_descs = st.multiselect(
'Select descriptors to visualize:',
desc_cols,
default=desc_cols[:min(4, len(desc_cols))]
)

if selected_descs:
fig = create_distribution_plots(desc_df, selected_descs)
st.pyplot(fig)
plt.close()

# Экспорт дескрипторов
st.download_button(
'📥 Download Descriptors (CSV)',
desc_df.to_csv(index=False),
'descriptors.csv',
'text/csv'
)

# Вкладка 3: Корреляционный анализ
with tabs[2]:
st.header('📈 Correlation Analysis')

# Выбор целевых переменных
target_cols = ['α·106 (K-1)', 'β', 'αav·106 (K-1)', 'T(bends), °C']
available_targets = [col for col in target_cols if col in df.columns]

if not available_targets:
st.warning('No target variables found in data.')
else:
selected_target = st.selectbox(
    'Select target variable:',
    available_targets
)

# Корреляционная матрица
st.subheader('Correlation Matrix')

# Выбор метода корреляции
corr_method = st.radio(
    'Correlation method:',
    ['pearson', 'spearman'],
    horizontal=True
)

# Подготовка данных для корреляции
numeric_cols = desc_df.select_dtypes(include=[np.number]).columns.tolist()
corr_data = desc_df[numeric_cols].dropna(axis=1, how='all')

if len(corr_data.columns) > 1:
    # Расчет корреляции
    corr_matrix, p_matrix = calculate_correlation_matrix(corr_data, method=corr_method)
    
    # Фильтрация по значимости
    threshold = st.slider('Correlation threshold:', 0.1, 0.9, 0.3)
    significant_mask = (np.abs(corr_matrix) > threshold) & (p_matrix < 0.05)
    corr_filtered = corr_matrix.where(significant_mask, 0)
    
    fig = create_correlation_matrix_plot(corr_filtered, 
                                         title=f'{corr_method.capitalize()} Correlation Matrix (|r| > {threshold}, p < 0.05)')
    st.pyplot(fig)
    plt.close()
    
    # Топ-дескрипторы
    st.subheader('Top Descriptors')
    
    top_results = find_top_descriptors(corr_data, [selected_target], n=20)
    if selected_target in top_results:
        st.dataframe(top_results[selected_target])
    
    # Сетевой граф
    st.subheader('Correlation Network')
    
    network_threshold = st.slider('Network correlation threshold:', 0.3, 0.9, 0.5)
    fig = create_correlation_network(corr_matrix, threshold=network_threshold)
    st.pyplot(fig)
    plt.close()
    
    # Partial correlation
    st.subheader('Partial Correlation')
    
    control_vars = st.multiselect(
        'Select control variables:',
        numeric_cols,
        default=['pH2O'] if 'pH2O' in numeric_cols else []
    )
    
    if control_vars and selected_target in corr_data.columns:
        partial_corr = calculate_partial_correlation(corr_data, selected_target, control_vars)
        st.dataframe(partial_corr.sort_values('correlation', key=abs, ascending=False))

# Вкладка 4: PCA и кластеризация
with tabs[3]:
st.header('🧬 PCA & Clustering Analysis')

# Подготовка данных
numeric_cols = desc_df.select_dtypes(include=[np.number]).columns.tolist()
pca_data, _ = prepare_pca_data(desc_df[numeric_cols], n_features=30)

if len(pca_data.columns) > 1:
# PCA
st.subheader('Principal Component Analysis')

n_components = st.slider('Number of components:', 2, 10, 5)
pca_result, pca, explained_var, cum_var = perform_pca(pca_data, n_components)

# Elbow plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(explained_var)+1), cum_var, 'bo-', linewidth=2)
ax.axhline(y=0.85, color='red', linestyle='--', label='85% variance')
ax.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Explained Variance', fontsize=12, fontweight='bold')
ax.set_title('PCA Explained Variance', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)
plt.close()

# Biplot
st.subheader('PCA Biplot')

# Выбор цвета для точек
color_options = ['None'] + [col for col in df.columns if col in ['method', 'B', 'A']]
color_col = st.selectbox('Color points by:', color_options)

labels = None
if color_col != 'None' and color_col in df.columns:
    # Используем категориальные данные для цвета
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels = le.fit_transform(df[color_col].fillna('Unknown').astype(str))

fig = create_pca_biplot(pca_result, pca, pca_data.columns, labels)
st.pyplot(fig)
plt.close()

# 3D PCA
st.subheader('3D PCA Visualization (Interactive)')

if pca_result.shape[1] >= 3:
    fig = create_pca_3d(pca_result[:, :3], labels)
    st.plotly_chart(fig)

# Кластеризация
st.subheader('Clustering Analysis')

cluster_method = st.radio(
    'Clustering method:',
    ['K-means', 'DBSCAN'],
    horizontal=True
)

if st.button('Run Clustering'):
    with st.spinner('Clustering...'):
        if cluster_method == 'K-means':
            labels, model, n_clusters, silhouette = perform_clustering(
                pca_data, method='kmeans'
            )
            st.success(f'K-means clustering completed: {n_clusters} clusters, silhouette score: {silhouette:.3f}')
        else:
            labels, model, eps, silhouette = perform_clustering(
                pca_data, method='dbscan'
            )
            st.success(f'DBSCAN clustering completed: {len(set(labels))} clusters, eps: {eps:.3f}')
        
        # Добавляем метки кластеров к данным
        df_clustered = df.copy()
        df_clustered['Cluster'] = labels
        
        # Визуализация кластеров на PCA
        st.subheader('Clusters on PCA')
        fig = create_pca_biplot(pca_result, pca, pca_data.columns, labels)
        st.pyplot(fig)
        plt.close()
        
        # t-SNE
        st.subheader('t-SNE Visualization')
        tsne_result, _ = perform_tsne(pca_data)
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                            c=labels, cmap='viridis', alpha=0.7)
        ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
        ax.set_title('t-SNE Projection with Clusters', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax)
        st.pyplot(fig)
        plt.close()
        
        # Cluster profiles
        st.subheader('Cluster Profiles')
        cluster_means = df_clustered.groupby('Cluster')[numeric_cols].mean()
        st.dataframe(cluster_means)
        
        # Heatmap of cluster profiles
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(cluster_means.T, cmap='coolwarm', annot=True, fmt='.2f', ax=ax)
        ax.set_title('Cluster Profiles', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()

# Вкладка 5: Визуализации
with tabs[4]:
st.header('📊 Advanced Visualizations')

# Выбор типа визуализации
viz_type = st.selectbox(
'Select visualization type:',
[
    'Concentration Heatmap',
    'Concentration Contour',
    'Bubble Chart (4D)',
    'Compositional Bubble',
    'Pairplot Top Features',
    'α vs β Compromise',
    'T(bends) Analysis',
    'Feature Importance',
    'Distribution Plots'
]
)

if viz_type == 'Concentration Heatmap':
st.subheader('Concentration Heatmap')

# Выбор осей
x_col = st.selectbox('X-axis:', desc_df.columns, index=0)
y_col = st.selectbox('Y-axis:', desc_df.columns, index=min(1, len(desc_df.columns)-1))
z_col = st.selectbox('Color (target):', 
                    ['α·106 (K-1)', 'β', 'αav·106 (K-1)', 'T(bends), °C'],
                    index=0)

if z_col in df.columns:
    fig = create_concentration_heatmap(df, x_col, y_col, z_col)
    if fig:
        st.plotly_chart(fig)
    else:
        st.warning('Insufficient data for heatmap.')

elif viz_type == 'Bubble Chart (4D)':
st.subheader('4D Bubble Chart')

x_col = st.selectbox('X-axis:', desc_df.columns, index=0)
y_col = st.selectbox('Y-axis (target):', 
                    ['α·106 (K-1)', 'β', 'αav·106 (K-1)', 'T(bends), °C'],
                    index=0)
color_col = st.selectbox('Color:', desc_df.columns, index=min(1, len(desc_df.columns)-1))
size_col = st.selectbox('Size:', desc_df.columns, index=min(2, len(desc_df.columns)-1))
shape_col = st.selectbox('Shape (optional):', 
                        ['None'] + [col for col in df.columns if col in ['method', 'B', 'A']],
                        index=0)

if y_col in df.columns and color_col in df.columns and size_col in df.columns:
    shape = None if shape_col == 'None' else shape_col
    fig = create_bubble_chart(df, x_col, y_col, color_col, size_col, shape)
    if fig:
        st.plotly_chart(fig)
    else:
        st.warning('Insufficient data for bubble chart.')

elif viz_type == 'Pairplot Top Features':
st.subheader('Pairplot of Top Features')

hue_col = st.selectbox('Color by (optional):', 
                      ['None'] + [col for col in df.columns if col in ['method', 'B', 'A']],
                      index=0)
n_features = st.slider('Number of features:', 2, 8, 5)

hue = None if hue_col == 'None' else hue_col
fig = create_pairplot_top_features(df, hue, n_features)
if fig:
    st.pyplot(fig)
    plt.close()
else:
    st.warning('Insufficient data for pairplot.')

elif viz_type == 'α vs β Compromise':
st.subheader('α vs β Compromise Diagram')

alpha_col = st.selectbox('Select α column:', 
                        [col for col in df.columns if 'α' in col.lower()],
                        index=0)
beta_col = st.selectbox('Select β column:', 
                       [col for col in df.columns if 'β' in col.lower()],
                       index=0)

if alpha_col in df.columns and beta_col in df.columns:
    fig = create_alpha_beta_compromise(df, alpha_col, beta_col)
    if fig:
        st.pyplot(fig)
        plt.close()
    else:
        st.warning('Insufficient data for α vs β plot.')

elif viz_type == 'T(bends) Analysis':
st.subheader('T(bends) vs δ Analysis')

delta_col = st.selectbox('Select δ column:', 
                        [col for col in df.columns if 'delta' in col.lower() or 'δ' in col],
                        index=0)
t_bends_col = st.selectbox('Select T(bends) column:', 
                          [col for col in df.columns if 'bends' in col.lower() or 'T(bends)' in col],
                          index=0)

if delta_col in df.columns and t_bends_col in df.columns:
    fig = create_t_bends_analysis(df, delta_col, t_bends_col)
    if fig:
        st.pyplot(fig)
        plt.close()
    else:
        st.warning('Insufficient data for T(bends) analysis.')

elif viz_type == 'Feature Importance':
st.subheader('Feature Importance Analysis')

target_col = st.selectbox('Select target variable:', 
                         ['α·106 (K-1)', 'β', 'T(bends), °C'],
                         index=0)

if target_col in df.columns:
    fig = create_feature_importance(df, target_col)
    if fig:
        st.pyplot(fig)
        plt.close()
    else:
        st.warning('Insufficient data for feature importance analysis.')

elif viz_type == 'Distribution Plots':
st.subheader('Distribution Plots')

selected_cols = st.multiselect(
    'Select columns for distribution plots:',
    desc_df.columns,
    default=desc_df.columns[:min(4, len(desc_df.columns))].tolist()
)

if selected_cols:
    fig = create_distribution_plots(desc_df, selected_cols)
    st.pyplot(fig)
    plt.close()

# Вкладка 6: Экспорт
with tabs[5]:
st.header('💾 Export Results')

st.subheader('Export Data')

# Экспорт полного DataFrame
st.download_button(
'📥 Download Full Data (CSV)',
df.to_csv(index=False),
'full_data.csv',
'text/csv'
)

# Экспорт дескрипторов
st.download_button(
'📥 Download Descriptors (CSV)',
desc_df.to_csv(index=False),
'descriptors.csv',
'text/csv'
)

# Экспорт статистики
st.subheader('Summary Statistics')
st.dataframe(df.describe())

# Информация о данных
st.subheader('Data Information')
col1, col2, col3 = st.columns(3)
with col1:
st.metric('Samples', len(df))
with col2:
st.metric('Features', len(df.columns))
with col3:
missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
st.metric('Missing Data', f'{missing_pct:.1f}%')

# ============================================================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================================================

if __name__ == '__main__':
main()
