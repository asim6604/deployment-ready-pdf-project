def chunking(reader):
    complete=""
    for i in range(len(reader.pages)):
        page=reader.pages[i]
        neet=page.extract_text()
        complete=complete+ " "+neet
    chunk=[]
    start=0
    end=500
    length=len(complete)
    iterate=int(length/450)
   
    while  start<length:
        
        chunk.append(complete[start:end])
        
        start=start+450
        
        end=end+450;
    return chunk




