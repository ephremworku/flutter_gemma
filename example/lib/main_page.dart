import 'package:flutter/material.dart';

import 'embedding_chat_screen.dart';
import 'model_selection_screen.dart';


class MainPageFor extends StatefulWidget {
  const MainPageFor({super.key});

  @override
  State<MainPageFor> createState() => _MainPageForState();
}

class _MainPageForState extends State<MainPageFor> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Main Page"),),
      body: Center(
        child: Column(
          children: [
            ElevatedButton(onPressed: (){
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const ModelSelectionScreen()),
              );
            }, child: const Text("chat with the model")),
            ElevatedButton(onPressed: (){
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const ChatScreenCustom()),
              );
            }, child: const Text("Text Embedding")),
          ],
        ),
      ),
    );
  }
}
