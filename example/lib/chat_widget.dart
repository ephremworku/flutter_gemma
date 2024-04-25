import 'package:flutter/material.dart';
import 'package:flutter_gemma_example/chat_input_field.dart';
import 'package:flutter_gemma_example/chat_message.dart';
import 'package:flutter_gemma_example/core/message.dart';
import 'package:flutter_gemma_example/gemma_input_field.dart';

class ChatListWidget extends StatelessWidget {
  const ChatListWidget({
    super.key,
    required this.messages,
    required this.gemmaHandler,
    required this.humanHandler,
  });

  final List<Message> messages;
  final ValueChanged<Message> gemmaHandler;
  final ValueChanged<String> humanHandler;

  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      padding: const EdgeInsets.all(8.0),
      reverse: true,
      itemCount: messages.length + 2,
      itemBuilder: (context, index) {
        if (index == 0) {
          if (messages.isNotEmpty && messages.last.isHuman) {
            return GemmaInputField(
              messages: messages,
              streamHandled: gemmaHandler,
            );
          }
          if (messages.isEmpty || !messages.last.isHuman) {
            return ChatInputField(handleSubmitted: humanHandler);
          }
        } else if (index == 1) {
          return const Divider(height: 1.0);
        } else {
          final message = messages.reversed.toList()[index - 2];
          return ChatMessageWidget(
            message: message,
          );
        }
      },
    );
  }
}
