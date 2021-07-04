# diplom_ml

В переменную path указать путь до рабочей папки
В переменную path_data указать путь до папки с подпапками: test, train_kaggle

Чтобы не обучать модель повторно, комментируем данные строки кода 
```
# Обучаем и сохраняем веса модели
mcp = ModelCheckpoint(path + 'mdl_wts.hdf5',\
                      save_best_only=True,\
                      monitor='accuracy')
opt = Adam(lr=0.0001, decay=1e-6)


model.compile(loss='categorical_crossentropy',\
              optimizer=opt,\
              metrics=['accuracy'])
model.fit(train_generator,\
          epochs=EPOCHS,\
          steps_per_epoch=STEPS_PER_EPOCH,\
          callbacks=[mcp])
```
