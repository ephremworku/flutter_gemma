import 'package:flutter/material.dart';
import 'package:flutter_gemma/flutter_gemma_interface.dart';

// import 'logic/native_bridge_code.dart';


class ChatScreenCustom extends StatefulWidget {
  const ChatScreenCustom({super.key});

  @override
  State<ChatScreenCustom> createState() => _ChatScreenCustomState();
}

class _ChatScreenCustomState extends State<ChatScreenCustom> {
  final TextEditingController _controller = TextEditingController();
  final List<Map<String, String>> _messages = [];
  final _embeddings = FlutterGemmaPlugin.instance;
  List<double>? _embedResponse;
  String _stringEmbedding = "";
  String _error = "";

  void _sendMessage() async {
    print('hello there');
    if (_controller.text.trim().isEmpty) return;
    try{
      _stringEmbedding = "${await _embeddings.getEmbeddingText(_controller.text)}";
      print(_stringEmbedding);

    }catch (e){
      print(e);
      _error = "$e";
    }
    // print("the embedding of:\n${_controller.text} is: \n$values");
    setState(() {
      _messages.add({'role': 'user', 'text': _controller.text.trim()});
      _messages.add({'role': 'ai', 'text': 'embedding: $_stringEmbedding, error: $_error'});
    });
    _controller.clear();
  }

  Widget _buildMessage(Map<String, String> message) {
    final isUser = message['role'] == 'user';
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 4, horizontal: 8),
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: isUser ? Colors.blueAccent : Colors.grey[300],
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(12),
            topRight: const Radius.circular(12),
            bottomLeft: Radius.circular(isUser ? 12 : 0),
            bottomRight: Radius.circular(isUser ? 0 : 12),
          ),
        ),
        child: Text(
          message['text']!,
          style: TextStyle(
            color: isUser ? Colors.white : Colors.black87,
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Chat")),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              itemCount: _messages.length,
              itemBuilder: (_, index) => _buildMessage(_messages[index]),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: InputDecoration(
                      hintText: "Type your message...",
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                      contentPadding: const EdgeInsets.symmetric(horizontal: 16),
                    ),
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.send),
                  onPressed: _sendMessage,
                  color: Theme.of(context).primaryColor,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
