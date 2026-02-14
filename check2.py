import uuid
def generate_thread_id():
    thread_id=uuid.uuid4()
    return thread_id

response=generate_thread_id()
print (type(response))
print (response)