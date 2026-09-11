# simul_translate_demo
1. demo_web_asr_semantic.py：标点+vad停顿，这两个条件同时满足才切；也就是vad辅助语义切分；问题是有些延后，感知上；确认tts来GPU加速是否已经实现
2. demo_web_asr_slience.py：静音停顿生成（1s增量），仅凭vad来断句
3. demo_web_asr_full_session5s.py:增量asr，增量生成；问题是增量asr出来的结果会不准；
4. demo_web_asr_clause.py是滑动窗口+时间戳+稳定切点；目前是响应较快

# polish 路线
1. 基于sensevoice+fsmn_vad+hunyuan+supertonic;
2. 整体围绕低时延的同声传译；主要是在asr和tts侧做工作；
3. 最终本方案是非流式asr+语义切分；按照逗号切分，或者句尾1.5svad；

asr侧
1. 语义切分/停顿切分
asr侧需要对语音做切分，语义切分/停顿切分；语义切分依靠asr生成的标点，需要模型有timestep，asr输出符号不准确导致切分错误；停顿切分依靠vad，句子过长停不下来，口头停顿语义不完整；

2. 流式/非流式asr
流式asr模型，类似paraformer，流式生成过程中没有标点生成，完整生成后利用ct_punc做完整标点标注；
非流式asr，例如本例子中使用的sensevoice，生成过程中会输出标点，但是增量asr的过程中会导致符号有修改，甚至是自动补标点；




tts侧，标点切分，分块生成，减少首包延迟




