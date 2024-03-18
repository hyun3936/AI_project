# 최초 모델 학습

from Model import Autoencoder
import os









# 모델 저장 경로 설정 ----------------------------------------------------------------------------------------
model_save_path = './best_model.pth'

# 초기 최소 손실 값을 매우 큰 값으로 설정
min_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # 에포크마다 평균 훈련 손실 계산
    train_loss = train_loss / len(train_loader)
    
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    
    # 에포크마다 평균 테스트 손실 계산
    test_loss = test_loss / len(test_loader)
    
    print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Test Loss: {test_loss}')
    
    # 현재 모델이 이전 모델보다 성능이 좋으면 저장
    if test_loss < min_loss:
        min_loss = test_loss
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved at Epoch {epoch+1} with Test Loss: {test_loss}')



