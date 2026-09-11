# simul_translate_demo
demo_web_asr_semantic.py：标点+vad停顿，这两个条件同时满足才切；也就是vad辅助语义切分；问题是有些延后，感知上；确认tts来GPU加速是否已经实现
demo_web_asr_slience.py：静音停顿生成（1s增量），仅凭vad来断句
demo_web_asr_full_session5s.py:增量asr，增量生成；问题是增量asr出来的结果会不准；
demo_web_asr_clause.py是滑动窗口+时间戳+稳定切点；目前是响应较快
