# Fake booking API

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd

from data_models import HotelData, AvailabilityRequest, AvailabilityResponse, BookingRequest, BookingResponse

hotels = pd.read_csv('hotel_db.csv')

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"message": "Invalid input", "errors": exc.errors()},
    )

@app.post('/get_hotels')
def get_hotels(request: AvailabilityRequest) -> AvailabilityResponse:
    db_response = hotels.query('city == @request.city')
    if len(db_response) == 0:
        return AvailabilityResponse(
            available_hotels=None,
            success=False,
            error_message='No available hotels in this city at these dates'
        )
    else:
        available_hotels = [HotelData(**hotel) for hotel in db_response.to_dict(orient='records')]
        return AvailabilityResponse(
            available_hotels=available_hotels
        )

@app.post('/book')
def book_hotel(request: BookingRequest) -> BookingResponse:
    availability = hotels.query('city == @request.city and name == @request.name')
    if len(availability) == 0:
        return BookingResponse(
            success=False,
            message='Sorry, no rooms available',
            **request.model_dump()
        )
    else:
        return BookingResponse(
            success=True,
            message='Your booking is confirmed',
            **request.model_dump()
        )


if __name__ == '__main__':
    uvicorn.run('main:app')