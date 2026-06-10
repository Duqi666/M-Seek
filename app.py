#!/usr/bin/env python3
"""
RAG 问答服务 - Flask 后端
部署到 Render/Replit 等平台
"""
import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get('DASHSCOPE_API_KEY', 'sk-37c32105b3674024a90a612856c6a4ad')

rag_service = None

def init_rag_service():
    global rag_service
    if rag_service is None:
        from rag import RAGService
        rag_service = RAGService()
    return rag_service

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'success': False, 'error': '缺少 question 参数'}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({'success': False, 'error': '问题不能为空'}), 400
        
        init_rag_service()
        
        docs = rag_service.retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt = rag_service.prompt.format(context=context, question=question)
        response = rag_service.chat_model.invoke(prompt)
        
        sources = []
        for doc in docs:
            source_info = {
                'source': doc.metadata.get('source', 'unknown'),
                'file_name': doc.metadata.get('file_name', 'unknown'),
                'create_time': str(doc.metadata.get('create_time', ''))
            }
            sources.append(source_info)
        
        return jsonify({
            'success': True,
            'answer': response.content,
            'sources': sources,
            'context_length': len(context)
        })
    
    except Exception as e:
        app.logger.error(f'Error processing request: {str(e)}')
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'RAG API'})

@app.route('/')
def index():
    return "<h1>RAG 问答服务</h1><p>API endpoint: POST /api/chat</p>"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)