# OpenSearch 2.17.1 + 中文 IK 分词器
# 镜像首次构建时会从 InfiniLabs 下载与 OpenSearch 同版本的 IK 插件,
# 安装后即支持 ik_max_word / ik_smart 两种 analyzer。
#
# 升级 OpenSearch 时务必同步升级 IK_VERSION (二者版本必须一致)。
FROM opensearchproject/opensearch:2.17.1

ARG IK_VERSION=2.17.1
ARG IK_URL=https://release.infinilabs.com/analysis-ik/stable/opensearch-analysis-ik-${IK_VERSION}.zip

USER root
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch "$IK_URL" \
    && chown -R opensearch:opensearch /usr/share/opensearch/plugins
USER opensearch
