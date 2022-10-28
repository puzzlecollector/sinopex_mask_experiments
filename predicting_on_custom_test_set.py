'''
necessary library imports 
'''
model = Classifier() 
checkpoint = torch.load("test_model_KFOLD1.pt") 
model.load_state_dict(checkpoint) 
model.cuda() 
model.eval() 
predictions = []
true_labels = [] 

for step, batch in tqdm(enumerate(val_dataloader), desc="validating", position=0, leave=True, total=len(val_dataloader)): 
    batch = tuple(t.to(device) for t in batch) 
    b_input_ids, b_input_masks, b_labels = batch 
    with torch.no_grad(): 
        outputs = model(b_input_ids, b_input_masks) 
    logits = outputs 
    loss = loss_func(logits, b_labels) 
    val_loss += loss.item() 
    logits_cpu = logits.detach().cpu().numpy()
    label_ids = b_labels.detach().cpu().numpy()  
    
    pred_labels = np.argmax(logits_cpu, axis=1) 
    for p in pred_labels:
        predictions.append(p) 
    for l in label_ids:
        true_labels.append(l) 
        
def predict_class(text): 
    encoded_input = tokenizer(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(device)
    input_ids = encoded_input["input_ids"] 
    attention_mask = encoded_input["attention_mask"] 
    

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    outputs = outputs.detach().cpu().numpy()

    predicted_class = np.argmax(outputs, axis=1) 
    return predicted_class[0]
  
test_data = ["비트코인 선물 투자 해볼까?",
             "카카오 주식 좇됐네",
             "아 심심해 오늘 이태원 갈까",
             "돈까스 좋아해?",
             "감기가 심해",
             "상품 불량품 관련 문의", 
             "프리미엄 라이트랑은 무슨 차이인가요?",
             "마스크 100매가 더 비싼 것 맞나요?",
             "사이즈 교환 문의요",
             "마스크 화이트 색상은 선택 못하나요?",
             "교환이나 반품을 하려고 합니다"] 

for t in test_data:
    print(predict_class(t))
